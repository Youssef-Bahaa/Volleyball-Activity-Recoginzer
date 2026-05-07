import torch
import torch.nn as nn
from src.models.B1.B1_model import ResNetFineTune
from src.utils.checkpoint import CheckpointManager

class B4Model(nn.Module):
    def __init__(self, b1_path, input_dim=512, hidden_dim=256, num_layers=2, num_classes=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Load frozen B1 as feature extractor
        cnn = ResNetFineTune(num_classes=num_classes)
        CheckpointManager.load(b1_path, cnn)
        cnn.backbone.fc = cnn.backbone.fc[:-1]  # strip final Linear: outputs 512
        for p in cnn.parameters():
            p.requires_grad = False
        self.cnn = cnn

        # Temporal model
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        with torch.no_grad():
            feats = self.cnn(x.view(B * T, C, H, W))   # (B*T, 512)
        feats = feats.view(B, T, -1)                    # (B, T, 512)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])                   # last time step: (B, 8)

