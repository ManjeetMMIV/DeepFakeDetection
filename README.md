DeepFake Detection Using ResNeXt-50 + LSTM
This project implements a deepfake video classification system using a combination of ResNeXt-50 (for spatial feature extraction) and LSTM (for temporal sequence modeling). The model is trained to distinguish between real and fake videos using the Celeb-DF (v1) dataset.

📁 Dataset
Source: Celeb-DF (v1)

Composition:

🟢 428 real videos

🔴 432 fake videos

⚙️ Preprocessing Pipeline
Video Trimming: Each video is trimmed to 150 frames using OpenCV.

Face Extraction: Faces are cropped from these frames. For each video, 10 valid face frames are selected.

Debugging Support: Videos with no detected faces are displayed for manual review.

Transformations: Selected face frames are resized and transformed to the input shape required for ResNeXt-50.

🧠 Model Architecture
CNN Backbone: Pretrained ResNeXt-50 extracts spatial features from individual frames.

Temporal Modeling: Features from 10 frames are fed into an LSTM to capture sequential patterns.

Classification Head: Final linear layer outputs class logits.

Loss Function: CrossEntropyLoss with class weighting to manage imbalance.

🏷️ Label Mapping
Real: 0

Fake: 1

🛠️ Frameworks & Libraries
PyTorch

OpenCV

📊 Training Results
Training Accuracy: 89.7%

Training Loss: 0.17

🧪 Testing Results
Testing Accuracy: 98.26%

Average Test Loss: 0.0703

🔍 Confusion Matrix
Predicted: Real (0)	Predicted: Fake (1)
Actual: Real (0)	86	2
Actual: Fake (1)	1	83
