import torch.nn
import torch.nn as nn

class Temporal_Group_Classifier(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_classes=8):
        super().__init__()

        self.lstm = nn.LSTM(hidden_size=512,
                            input_size=2048,
                            batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)
