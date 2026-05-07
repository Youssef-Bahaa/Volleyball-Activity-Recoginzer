import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataset.transforms import train_transform, val_transform, test_transform
from src.dataset.utils import activity2id, TransformSubset, filter_by_ids
import sys
import src.dataset.boxinfo as boxinfo

sys.modules["boxinfo"] = boxinfo


class VolleyBallClipDataset(Dataset):
    def __init__(self, image_root, annot_path):
        self.samples = []

        with open(annot_path, 'rb') as f:
            data = pickle.load(f)

        for video_id in os.listdir(image_root):
            video_dir = os.path.join(image_root, video_id)
            if not os.path.isdir(video_dir):
                continue

            for clip_id in os.listdir(video_dir):
                clip_dir = os.path.join(video_dir, clip_id)
                if not os.path.isdir(clip_dir):
                    continue

                try:
                    clip_info = data[video_id][clip_id]
                    activity = activity2id(clip_info['category'])
                    frame_boxes = clip_info['frame_boxes_dct']
                except KeyError:
                    continue

                sorted_frames = sorted(frame_boxes.keys())
                img_paths = [
                    os.path.join(clip_dir, f'{fid}.jpg')
                    for fid in sorted_frames
                    if os.path.exists(os.path.join(clip_dir, f'{fid}.jpg'))
                ]

                if not img_paths:
                    continue

                self.samples.append((video_id, clip_id, img_paths, activity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, clip_id, img_paths, label = self.samples[idx]
        imgs = [Image.open(p).convert('RGB') for p in img_paths]
        return video_id, clip_id, imgs, torch.tensor(label, dtype=torch.long)


def build_loaders(cfg):
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    full_dataset = VolleyBallClipDataset(
        image_root = data_cfg["videos_path"],
        annot_path = data_cfg["annot_path"]
    )

    train_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["train"])
    val_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["validation"])
    test_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["test"])

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)
    test_dataset = TransformSubset(test_subset, test_transform)

    loader_kwargs = {
        "batch_size": training_cfg["batch_size"],
        "num_workers": training_cfg.get("num_workers", 4),
        "pin_memory": training_cfg.get("pin_memory", True),
    }

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader