import torch
from torch.utils.data import DataLoader, Subset, Dataset
import os
import pickle
import cv2
from PIL import Image

from src.dataset.transforms import train_transform, val_transform
from src.dataset.utils import activity2id, TransformSubset, filter_by_ids, PERSON_ACTION_TO_ID
from src.dataset.boxinfo import BoxInfo


class PersonDataset(Dataset):
    """
    Phase 1 dataset — one sample per player crop per clip.
    Returns (video_id, crop_tensor, action_label)
    """
    def __init__(self, image_root, annot_path):
        self.samples = []

        with open(annot_path, 'rb') as f:
            data = pickle.load(f)

        for video_id in os.listdir(image_root):
            video_dir = os.path.join(image_root, video_id)
            if not os.path.isdir(video_dir):
                continue

            for clip in os.listdir(video_dir):
                clip_dir   = os.path.join(video_dir, clip)
                if not os.path.isdir(clip_dir):
                    continue

                frame_path = os.path.join(clip_dir, f'{clip}.jpg')
                try:
                    boxes = data[video_id][clip]['frame_boxes_dct'][int(clip)]
                except KeyError:
                    continue

                for box in boxes:
                    self.samples.append((video_id, frame_path, box, box.action))

    def __len__(self,):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, frame_path, box, action = self.samples[idx]

        img = cv2.imread(frame_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {frame_path}")

        img = Image.fromarray(img[:, :, ::-1])  # BGR to RGB
        cropped = box.crop_from_frame(img)

        return video_id, cropped, PERSON_ACTION_TO_ID[action]




def build_loaders(cfg):
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    full_dataset = PersonDataset(
        data_cfg["videos_path"],
        data_cfg["annot_path"]
    )

    train_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["train"])
    val_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["validation"])
    test_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["test"])

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)
    test_dataset = TransformSubset(test_subset, val_transform)


    loader_kwargs = {
        "batch_size":  training_cfg["batch_size"],
        "num_workers": training_cfg.get("num_workers", 4),
        "pin_memory":  training_cfg.get("pin_memory", True),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

