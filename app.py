import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import face_recognition
from PIL import Image, ImageOps
import tempfile
import os
import numpy as np

# Import the model architecture
from model_definition import DeepfakeModel

# 1. Load Model Helper
@st.cache_resource
def load_model(weights_path, quantized=False):
    model = DeepfakeModel()
    if quantized:
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

# 2. Extract a single segment of face-cropped frames from a video
def extract_segment(video_path, start_frame, num_frames=10, frame_stride=15):
    """Extract num_frames face-cropped frames starting from start_frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames = []
    face_box = None
    first_cropped_pil = None

    for i in range(num_frames):
        frame_idx = start_frame + i * frame_stride
        if frame_idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face on first frame of segment, reuse box for rest
        if face_box is None:
            locs = face_recognition.face_locations(rgb, model="hog")
            if locs:
                face_box = locs[0]

        if face_box:
            top, right, bottom, left = face_box
            top = max(0, top)
            left = max(0, left)
            bottom = min(rgb.shape[0], bottom)
            right = min(rgb.shape[1], right)
            crop = rgb[top:bottom, left:right]
            if crop.size == 0:
                crop = rgb
        else:
            crop = rgb

        pil_img = Image.fromarray(crop)

        if face_box is not None and first_cropped_pil is None:
            first_cropped_pil = pil_img

        frames.append(transform(pil_img))

    cap.release()

    if len(frames) == 0:
        frames = [torch.zeros(3, 112, 112)] * num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames).unsqueeze(0), first_cropped_pil  # [1, T, C, H, W]


def run_tta_inference(model, segment_tensor):
    """Run model on the original segment + its horizontal flip, average the probabilities."""
    with torch.no_grad():
        # Original
        _, out_orig = model(segment_tensor)
        prob_orig = torch.softmax(out_orig, dim=1)[0]

        # Horizontally flip all frames: flip the W dimension (dim=4 in [1,T,C,H,W])
        flipped = torch.flip(segment_tensor, dims=[4])
        _, out_flip = model(flipped)
        prob_flip = torch.softmax(out_flip, dim=1)[0]

        # Average probabilities
        avg_prob = (prob_orig + prob_flip) / 2.0
    return avg_prob


def multi_segment_inference(model, video_path, num_segments=5, num_frames=10, frame_stride=15):
    """
    Split the video into num_segments equally-spaced starting points.
    Run the model (with TTA) on each segment.
    Average all the probabilities for a final robust prediction.
    Returns: (avg_real_prob, avg_fake_prob, per_segment_details, face_preview)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    segment_length = (num_frames - 1) * frame_stride + 1  # frames needed per segment

    # Calculate evenly-spaced start points
    if total_frames <= segment_length:
        # Video is too short for multiple segments — just use one
        start_points = [0]
    else:
        usable = total_frames - segment_length
        if num_segments == 1:
            start_points = [0]
        else:
            step = usable / (num_segments - 1)
            start_points = [int(i * step) for i in range(num_segments)]

    all_probs = []
    segment_details = []
    face_preview = None

    for idx, start in enumerate(start_points):
        seg_tensor, seg_face = extract_segment(video_path, start, num_frames, frame_stride)

        if face_preview is None and seg_face is not None:
            face_preview = seg_face

        avg_prob = run_tta_inference(model, seg_tensor)
        all_probs.append(avg_prob)

        segment_details.append({
            "segment": idx + 1,
            "start_frame": start,
            "real_prob": avg_prob[0].item() * 100,
            "fake_prob": avg_prob[1].item() * 100,
        })

    # Average all segment probabilities
    stacked = torch.stack(all_probs)
    final_prob = stacked.mean(dim=0)

    return final_prob[0].item() * 100, final_prob[1].item() * 100, segment_details, face_preview


# ─── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Deepfake Detector", layout="wide", initial_sidebar_state="expanded")

# Hide Streamlit UI elements for a cleaner, professional look and reduce top padding
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1rem;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #2e6c80; margin-top: 0;'>Deepfake Video Analysis Engine</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Powered by EfficientNet-B4 + Bi-LSTM</p>", unsafe_allow_html=True)
st.markdown("---")

# Default analysis config values
threshold = 45.0
num_segments = 5

# Model config
MODEL_PATH = "model_best (2).pth"
IS_QUANTIZED = False

if not os.path.exists(MODEL_PATH) and os.path.exists("../model_best.pth"):
    MODEL_PATH = "../model_best.pth"

if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Weights file not found in deployment folder.")
    st.info("Please place your downloaded model weights file inside the `deployment/` directory.")
else:
    try:
        model = load_model(MODEL_PATH, quantized=IS_QUANTIZED)
        # Main layout columns (balanced width)
        main_col_left, main_col_right = st.columns([1, 1])

        with main_col_left:
            st.success("Model weights loaded successfully.")
            st.write("### Upload Video")
            uploaded_file = st.file_uploader("Upload an MP4, AVI, or MOV video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")
            
            st.write("---")
            st.write("### Sample Videos")
            
            # Initialize session state to remember the selected video across button clicks
            if 'selected_video' not in st.session_state:
                st.session_state.selected_video = None
                
            sample_dir = "sample_videos"
            
            if os.path.exists(sample_dir):
                col1, col2 = st.columns(2)
                
                # Real videos
                with col1:
                    st.write("**Real Videos**")
                    real_dir = os.path.join(sample_dir, "Real")
                    if os.path.exists(real_dir):
                        real_vids = [f for f in os.listdir(real_dir) if f.endswith(('.mp4', '.avi', '.mov'))][:7]
                        if real_vids:
                            for vid in real_vids:
                                vid_path = os.path.join(real_dir, vid)
                                is_selected = (st.session_state.selected_video == vid_path)
                                label = f"Selected: {os.path.splitext(vid)[0]}" if is_selected else f"{os.path.splitext(vid)[0]}"
                                if st.button(label, key=f"real_{vid}", use_container_width=True):
                                    st.session_state.selected_video = vid_path
                        else:
                            st.info("Place `.mp4` files in:\n`sample_videos/Real`")
                            
                # Fake videos
                with col2:
                    st.write("**Fake Videos**")
                    fake_dir = os.path.join(sample_dir, "Fake")
                    if os.path.exists(fake_dir):
                        fake_vids = [f for f in os.listdir(fake_dir) if f.endswith(('.mp4', '.avi', '.mov'))][:7]
                        if fake_vids:
                            for vid in fake_vids:
                                vid_path = os.path.join(fake_dir, vid)
                                is_selected = (st.session_state.selected_video == vid_path)
                                label = f"Selected: {os.path.splitext(vid)[0]}" if is_selected else f"{os.path.splitext(vid)[0]}"
                                if st.button(label, key=f"fake_{vid}", use_container_width=True):
                                    st.session_state.selected_video = vid_path
                        else:
                            st.info("Place `.mp4` files in:\n`sample_videos/Fake`")

            # Determine which video to process
            video_to_process = None
            is_temp = False

            if uploaded_file is not None:
                # If user uploads a new file, clear the sample video selection
                st.session_state.selected_video = None
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()
                video_to_process = tfile.name
                is_temp = True
            elif st.session_state.selected_video:
                video_to_process = st.session_state.selected_video

        analyze_btn = False
        num_segments = 5
        
        # Right column: Dedicated entirely to the Video Player and Controls
        with main_col_right:
            st.write("### Source Video")
            if video_to_process:
                st.video(video_to_process)
                
                st.write("---")
                st.markdown("**⚙️ Analysis Settings**")
                num_segments = st.slider("Video Segments", min_value=1, max_value=10, value=5, step=1)
                st.caption("How it works: We extract frames from equally spaced segments across the video. Each segment is analyzed twice (original + flipped) using Test-Time Augmentation. All predictions are averaged for a highly robust final verdict.")
                
                st.write("---")
                analyze_btn = st.button("Start Deepfake Analysis (scroll down for results)", use_container_width=True, type="primary")
            else:
                st.info("Select a sample video or upload a file to preview it here.")
                
        # Full width block for results
        if video_to_process and analyze_btn:
                with st.spinner(f"Analyzing {num_segments} segment(s) with Test-Time Augmentation..."):
                    try:
                        real_prob, fake_prob, details, face_img = multi_segment_inference(
                            model, video_to_process, num_segments=num_segments
                        )

                        st.markdown("### Analysis Results")
                        
                        # Metrics Row
                        prediction = 1 if fake_prob >= threshold else 0
                        
                        res_col1, res_col2, res_col3 = st.columns(3)
                        with res_col1:
                            if prediction == 1:
                                st.metric(label="Verdict", value="DEEPFAKE")
                            else:
                                st.metric(label="Verdict", value="REAL")
                        with res_col2:
                            st.metric(label="Fake Confidence", value=f"{fake_prob:.1f}%")
                        with res_col3:
                            st.metric(label="Real Confidence", value=f"{real_prob:.1f}%")
                            
                        # Progress bar representing fake probability
                        if prediction == 1:
                            st.error("**High Probability of Manipulation Detected**")
                        else:
                            st.success("**Video Appears Authentic**")
                            
                        st.progress(fake_prob / 100.0, text="Manipulation Probability Score")

                        st.markdown("---")
                        
                        # Face tracking and technical breakdown in columns
                        tech_col1, tech_col2 = st.columns([1, 2])
                        with tech_col1:
                            st.markdown("**AI Face Tracking Scan**")
                            if face_img:
                                st.image(face_img, width=120, caption="Extracted Subject")
                                
                        with tech_col2:
                            with st.expander("Show Temporal Segment Breakdown", expanded=False):
                                st.markdown("Temporal analysis across video segments:")
                                for d in details:
                                    verdict = "Fake" if d["fake_prob"] >= threshold else "Real"
                                    st.write(f"- **Seg {d['segment']}** (frame {d['start_frame']}): Real {d['real_prob']:.1f}% | Fake {d['fake_prob']:.1f}% → {verdict}")

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                    finally:
                        if is_temp:
                            os.unlink(video_to_process)
                            
                        # Auto-scroll to the bottom of the page to show results
                        components.html(
                            """
                            <script>
                                const main = window.parent.document.querySelector('.main');
                                if (main) {
                                    main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
                                }
                            </script>
                            """,
                            height=0
                        )

    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()
