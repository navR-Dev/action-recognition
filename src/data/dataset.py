import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):

    def __init__(self, root):

        self.samples = []
        self.classes = sorted(os.listdir(root))

        for label, cls in enumerate(self.classes):
            folder = os.path.join(root, cls)

            for f in os.listdir(folder):
                path = os.path.join(folder, f)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        clip = np.load(path)
        clip = torch.tensor(clip, dtype=torch.float32)
        clip = clip.permute(3,0,1,2)  # C,T,H,W

        return clip, label