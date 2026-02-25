import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from extract_features import activity2id

class VolleyBallFeaturesDataset(Dataset):
    def __init__(self, feature_root):
        #path = \features\image-level\resnet\video_id\clip_id
        with open('annot_all.pkl' , 'rb') as f:
            data = pickle.load(f)

        self.samples = []

        for video_id in os.listdir(feature_root):
            video_dir = os.path.join(feature_root,video_id)

            for clip in os.listdir(video_dir):
                file_path = os.path.join(video_dir,f'{clip}.npy')
                activity  = data[video_id][clip]['category']
                activity = activity2id(activity)
                self.samples.append((file_path,activity))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
         clip_path , label = self.samples[idx]
         features = np.load(clip_path)
         # now getting the middle frame idx = 4
         frame5 = features[4]
         frame5 = torch.tensor(frame5 , dtype= torch.float32)
         label = torch.tensor(label , dtype = torch.long)
         return frame5 , label




