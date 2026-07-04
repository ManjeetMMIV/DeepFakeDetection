import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeModel(nn.Module):
    """
    This architecture EXACTLY matches deepfake-detection-optimized.ipynb (Cell 9 - v3).
    Backbone upgraded from ResNeXt50 to EfficientNet-B4 for superior deepfake artifact detection.
    
    Matches training notebook:
      - self.cnn  = backbone.features (EfficientNet-B4)
      - self.avgpool = AdaptiveAvgPool2d(1)
      - self.lstm  = Bidirectional LSTM, input_size=1792, hidden_size=512
      - self.classifier = Dropout(0.5) + Linear(1024, 2)
    """
    def __init__(self, num_classes=2, lstm_layers=1, hidden_dim=512, bidirectional=True):
        super(DeepfakeModel, self).__init__()

        # EfficientNet-B4: compound-scaled backbone, output 1792 channels
        backbone = models.efficientnet_b4(weights=None)
        self.cnn     = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        latent_dim = 1792  # EfficientNet-B4 output channels
        lstm_out   = hidden_dim * 2 if bidirectional else hidden_dim

        self.lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_out, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x    = x.view(B * T, C, H, W)
        fmap = self.cnn(x)
        feat = self.avgpool(fmap).view(B, T, 1792)
        out, _ = self.lstm(feat)
        out    = self.classifier(out.mean(1))
        return fmap, out
