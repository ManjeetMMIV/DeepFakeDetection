_**DeepFake Detection Using ResNeXt-50 + LSTM
This project implements a deepfake video classification system using a combination of ResNeXt-50 (for spatial feature extraction) and LSTM (for temporal sequence modeling). The model is trained to distinguish between real and fake videos using the Celeb-DF (v1) dataset.**
**_
📁 Dataset**
Source: Celeb-DF (v1)

**Composition:**
🟢 428 real videos
🔴 432 fake videos

**⚙️ Preprocessing Pipeline**
Video Trimming: Each video is trimmed to 150 frames using OpenCV.

Face Extraction: Faces are cropped from these frames. For each video, 10 valid face frames are selected.

Debugging Support: Videos with no detected faces are displayed for manual review.

Transformations: Selected face frames are resized and transformed to the input shape required for ResNeXt-50.

**🧠 Model Architecture**
CNN Backbone: Pretrained ResNeXt-50 extracts spatial features from individual frames.
Temporal Modeling: Features from 10 frames are fed into an LSTM to capture sequential patterns.
Classification Head: Final linear layer outputs class logits.
Loss Function: CrossEntropyLoss with class weighting to manage imbalance.

🏷️ Label Mapping
Real: 0

Fake: 1

🛠️ **Frameworks & Libraries:**
  1.PyTorch
  2.OpenCV

📊 **Training Results**
**Training Accuracy**__: **89.23%**

**Training Avg Loss: 0.1916**

🧪 **Testing Results**
**(Evaluation Score) Recall**: **86%**


🔍** Confusion Matrix**
<img width="624" height="580" alt="image" src="https://github.com/user-attachments/assets/cace4ac8-5e69-4ba5-b742-905c5cc611d1" />
**_Classification Report:
                    precision    recall  f1-score   support

  Real (Class 0)       0.88      0.98      0.92        88
Deepfake (Class 1)     0.97      0.86      0.91        84

        accuracy                           0.92       172
       macro avg       0.92      0.92      0.92       172
    weighted avg       0.92      0.92      0.92       172
    The most important evaluation score for this model the recall is found out to be 86%_**


