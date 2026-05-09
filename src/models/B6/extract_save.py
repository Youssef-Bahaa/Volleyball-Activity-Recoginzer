import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import pickle

from src.dataset.transforms import train_transform, val_transform
import src.dataset.boxinfo as _boxinfo_mod
from src.dataset.boxinfo import BoxInfo

import sys
sys.modules.setdefault('boxinfo', _boxinfo_mod)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "BoxInfo":
            return BoxInfo
        return super().find_class(module, name)


def extract_and_save(extractor, device, path, cfg):
    feature_extractor = extractor.get_feature_extractor().to(device)
    feature_extractor.eval()

    videos_root = str(path.videos)
    out_root = str(path.result("features_resnet"))

    train_ids = {str(i) for i in cfg["data"]["video_splits"]["train"]}
    annot_path = cfg["data"]["annot_path"]

    with open(annot_path, "rb") as f:
        data = _Unpickler(f).load()

    with torch.no_grad():
        for video_id in tqdm(os.listdir(videos_root), desc="Videos"):
            video_dir = os.path.join(videos_root, video_id)
            if not os.path.isdir(video_dir):
                continue

            for clip in tqdm(os.listdir(video_dir), desc=f"Clips {video_id}", leave=False):
                clip_dir = os.path.join(video_dir, clip)
                if not os.path.isdir(clip_dir):
                    continue

                try:
                    boxes = data[video_id][clip]["frame_boxes_dct"][int(clip)]
                except KeyError:
                    continue

                frame_path = os.path.join(clip_dir, f"{clip}.jpg")
                image = Image.open(frame_path).convert("RGB")

                transform = train_transform if video_id in train_ids else val_transform

                crops = torch.cat([
                    transform(image.crop(box.box)).unsqueeze(0)
                    for box in boxes
                ]).to(device)

                feats = feature_extractor(crops).squeeze(-1).squeeze(-1)  # (N, 2048)
                N = feats.shape[0]

                # Pad or truncate to 12 players
                if N < 12:
                    padding = torch.zeros(12 - N, feats.shape[1], device=feats.device)
                    feats = torch.cat([feats, padding], dim=0)
                else:
                    feats = feats[:12]

                # Aggregate (max over players)
                feats = torch.max(feats, dim=0).values  # (2048)

                out_file = os.path.join(out_root, video_id, f"{clip}.npy")
                os.makedirs(os.path.dirname(out_file), exist_ok=True)

                np.save(out_file, feats.cpu().numpy())