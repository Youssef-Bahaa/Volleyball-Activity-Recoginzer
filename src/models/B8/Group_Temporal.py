import torch
import torch.nn as nn

class GroupActivityB8(nn.Module):
    def __init__(self, player_model, hidden_size=512, num_classes=8):
        super(GroupActivityB8, self).__init__()

        base = player_model.module if hasattr(player_model, 'module') else player_model

        self.resnet50 = base.backbone
        self.lstm1 = base.lstm

        for param in self.resnet50.parameters():
            param.requires_grad = False
        for param in self.lstm1.parameters():
            param.requires_grad = False

        feat_dim = 2048 + hidden_size  # 2560 per player


        self.layer_norm_person = nn.LayerNorm(2048)          # before lstm1
        self.layer_norm_group  = nn.LayerNorm(feat_dim * 2)  # after cat(team1,team2)

        # pool players only, keep all features
        self.pool = nn.AdaptiveMaxPool1d(1)

        # two teams concatenated: 2560 * 2 = 5120
        self.lstm2 = nn.LSTM(
            input_size=feat_dim * 2,   # 5120
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        b, n, t, c, h, w = x.shape

        x = x.view(b * n * t, c, h, w)
        x = self.resnet50(x)                                      # (b*n*t, 2048, 1, 1)

        x = x.view(b * n, t, -1)                                  # (b*n, t, 2048)
        x = self.layer_norm_person(x)                              # normalize resnet feats
        out, _ = self.lstm1(x)                                     # (b*n, t, 512)

        x = torch.cat([x, out], dim=2).contiguous()               # (b*n, t, 2560)

        x = x.view(b * t, n, -1)                                  # (b*t, 12, 2560)
        team1 = x[:, :6, :]                                        # (b*t, 6, 2560)
        team2 = x[:, 6:, :]                                        # (b*t, 6, 2560)

        # pool 6 players -> 1, keep all 2560 features
        team1 = self.pool(team1.permute(0, 2, 1)).squeeze(-1)     # (b*t, 2560)
        team2 = self.pool(team2.permute(0, 2, 1)).squeeze(-1)     # (b*t, 2560)

        x = torch.cat([team1, team2], dim=1)                       # (b*t, 5120)
        x = x.view(b, t, -1)                                       # (b, t, 5120)
        x = self.layer_norm_group(x)                               # normalize group feats

        x, _ = self.lstm2(x)                                       # (b, t, 512)
        x = x[:, -1, :]                                            # (b, 512)

        return self.classifier(x)