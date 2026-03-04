import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):

    def __init__(self, root):

        self.samples = []

        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        for label, cls in enumerate(self.classes):

            folder = os.path.join(root, cls)

            for f in os.listdir(folder):
                if f.endswith(".npy"):

                    path = os.path.join(folder, f)

                    clips = np.load(path)  # load fully (dataset is now small)

                    for i in range(clips.shape[0]):
                        self.samples.append((path, i, label))

        print(
            f"Loaded {len(self.samples)} samples "
            f"from {len(self.classes)} classes."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, clip_idx, label = self.samples[idx]

        clips = np.load(path)
        clip = clips[clip_idx]   # (T, H, W, 2)

        clip = torch.tensor(clip, dtype=torch.float16)
        clip = clip.permute(3, 0, 1, 2)  # (2, T, H, W)

        return clip, label