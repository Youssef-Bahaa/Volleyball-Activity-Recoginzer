import torch.nn as nn

class B1FineTune(nn.Module):
    def __init__(self, in_dim=2048, num_classes=8):
        super(B1FineTune, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)

        )

    def forward(self, x):
        return self.classifier(x)
