import torch.nn
import torch.nn as nn
import torchvision.models as models

class B6Extractor(nn.Module):
    """ResNet50 fine-tuned on 9 person actions. Used in Phase 1 training + Phase 2 extraction."""
    def __init__(self, num_classes=9, pretrained=True):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_feature_extractor(self):
        """Returns backbone without the FC head — used for feature extraction."""
        return nn.Sequential(*list(self.backbone.children())[:-1])