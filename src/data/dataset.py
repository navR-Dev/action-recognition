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
                if not f.endswith(".npy"):
                    continue
                path = os.path.join(folder, f)
                clips = np.load(path, mmap_mode="r")
                num_clips = clips.shape[0]
                for i in range(num_clips):
                    self.samples.append((path, i, label))
        print(
            f"Loaded {len(self.samples)} samples "
            f"from {len(self.classes)} classes."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, clip_idx, label = self.samples[idx]
        clips = np.load(path, mmap_mode="r")
        clip = clips[clip_idx].copy()
        clip = torch.from_numpy(clip).float()
        clip = clip.permute(3, 0, 1, 2)  # (T,H,W,2) → (2,T,H,W)
        return clip, label