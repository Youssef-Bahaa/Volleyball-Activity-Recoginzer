import sys
import os
import cv2
import pickle

import src.dataset.boxinfo as _boxinfo_mod
sys.modules.setdefault('boxinfo', _boxinfo_mod)

from src.dataset.boxinfo import BoxInfo


def _is_kaggle():
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def get_paths():
    if _is_kaggle():
        dataset_root = '/kaggle/input/group-activity-recognition-volleyball'
        return {
            'videos': f'{dataset_root}/videos',
            'annot':  f'{dataset_root}/volleyball_tracking_annotation',
            'save':   '/kaggle/working/annot_all.pkl'
        }
    else:
        root = os.path.dirname(os.path.abspath(__file__))
        dataset_root = os.path.join(root, '..', '..', 'data')
        return {
            'videos': os.path.join(dataset_root, 'videos_dataset'),
            'annot':  os.path.join(dataset_root, 'volleyball_tracking_annotation'),
            'save':   os.path.join(root, 'annot_all.pkl')
        }


def load_frames_boxes(path):
    player_boxes = {i: [] for i in range(12)}
    frame_boxes = {}

    with open(path, 'r') as file:
        for line in file:
            player_box = BoxInfo(line)
            if player_box.player_ID > 11:
                continue
            player_boxes[player_box.player_ID].append(player_box)

        for Player_id, boxes_info in player_boxes.items():
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes:
                    frame_boxes[box_info.frame_ID] = []
                frame_boxes[box_info.frame_ID].append(box_info)

    return frame_boxes


def load_video_annot(video_path):
    clip_category = {}
    with open(os.path.join(video_path, 'annotations.txt'), 'r') as file:
        for line in file:
            seq = line.split()
            frame_id = seq[0].replace('.jpg', '')
            activity = seq[1]
            clip_category[frame_id] = activity
    return clip_category


def load_volleyball_dataset(videos_root, annot_root):
    videos_annot = {}

    for video_dir in sorted(os.listdir(videos_root)):
        video_dir_path = os.path.join(videos_root, video_dir)
        if not os.path.isdir(video_dir_path):
            continue

        clips_annot = {}
        activities = load_video_annot(video_dir_path)

        print(f'Processing Video {video_dir} / {len(os.listdir(videos_root))}')

        clips = [c for c in sorted(os.listdir(video_dir_path))
                 if os.path.isdir(os.path.join(video_dir_path, c))]

        for clip_dir in clips:
            annotation_path = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')

            if not os.path.exists(annotation_path):
                continue

            frame_boxes  = load_frames_boxes(annotation_path)
            clip_activity = activities.get(clip_dir)

            if clip_activity is None:
                continue

            clips_annot[clip_dir] = {
                'category':        clip_activity,
                'frame_boxes_dct': frame_boxes
            }

        videos_annot[video_dir] = clips_annot

    return videos_annot


def save_dataset():
    paths = get_paths()

    print(f"Videos : {paths['videos']}")
    print(f"Annot  : {paths['annot']}")
    print(f"Save to: {paths['save']}")

    videos_annot_dct = load_volleyball_dataset(paths['videos'], paths['annot'])

    os.makedirs(os.path.dirname(paths['save']), exist_ok=True)
    with open(paths['save'], 'wb') as f:
        pickle.dump(videos_annot_dct, f)

    print(f"Saved to {paths['save']}")
    return paths['save']


def load_dataset():
    paths = get_paths()
    with open(paths['save'], 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    save_dataset()