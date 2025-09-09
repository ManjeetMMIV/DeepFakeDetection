_**DeepFake Detection Using ResNeXt-50 + LSTM
This project implements a deepfake video classification system using a combination of ResNeXt-50 (for spatial feature extraction) and LSTM (for temporal sequence modeling). The model is trained to distinguish between real and fake videos using the Celeb-DF (v1) dataset.**__


📁 **Dataset**
Source: Celeb-DF (v1)


**Composition:**
🟢 428 real videos
🔴 432 fake videos

**⚙️ Preprocessing Pipeline**
_Video Trimming_: Each video is trimmed to 150 frames using OpenCV.

_Face Extraction_: Faces are cropped from these frames. For each video, 10 valid face frames are selected.

_Debugging Support_: Videos with no detected faces are displayed for manual review.

_Transformations_: Selected face frames are resized and transformed to the input shape required for ResNeXt-50.


**🧠 Model Architecture**
**CNN Backbone:** _Pretrained ResNeXt-50 extracts spatial features from individual frames._
**Temporal Modeling**: _Features from 10 frames are fed into an LSTM to capture sequential patterns._
**Classification Head**: _Final linear layer outputs class logits._
**Loss Function**:_ CrossEntropyLoss with class weighting to manage imbalance._

🏷️**Label Mapping**

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


🔍**Confusion Matrix**
<img width="624" height="580" alt="image" src="https://github.com/user-attachments/assets/cace4ac8-5e69-4ba5-b742-905c5cc611d1" />

_**Classification Report**_
<img width="729" height="277" alt="image" src="https://github.com/user-attachments/assets/1fb35121-e02b-4065-98b0-3b4f23188200" />

 **The most important evaluation score for this model - Recall is found out to be 86%**


