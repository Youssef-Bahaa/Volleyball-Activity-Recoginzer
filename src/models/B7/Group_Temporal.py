import torch
import torch.nn as nn

class GroupActivityB7(nn.Module):
    def __init__(self, player_model, hidden_size=512, num_classes=8):
        super(GroupActivityB7, self).__init__()

        base = player_model.module if hasattr(player_model, 'module') else player_model

        self.resnet50 = base.backbone
        self.lstm1 = base.lstm

        # freeze pretrained parts
        for param in self.resnet50.parameters():
            param.requires_grad = False

        for param in self.lstm1.parameters():
            param.requires_grad = False

        # normalization
        self.layer_norm_input = nn.LayerNorm(2048)
        self.layer_norm_feat = nn.LayerNorm(2048 + hidden_size)
        self.layer_norm_pool = nn.LayerNorm(2048 + hidden_size)

        # second LSTM
        self.lstm2 = nn.LSTM(
            input_size=2048 + hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        b, n, t, c, h, w = x.shape

        x = x.view(b * n * t, c, h, w)
        x = self.resnet50(x)

        x = x.view(b * n, t, -1)
        x = self.layer_norm_input(x)

        out, _ = self.lstm1(x)

        x = torch.cat([x, out], dim=2)  # (2048 + hidden)
        x = self.layer_norm_feat(x)

        x = x.view(b, n, t, -1)        # (b, players, time, features)
        x = x.permute(0, 2, 1, 3)      # (b, time, players, features)

        x = torch.max(x, dim=2)[0]


        x = self.layer_norm_pool(x)

        # ---- LSTM 2 ----
        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        return self.classifier(x)