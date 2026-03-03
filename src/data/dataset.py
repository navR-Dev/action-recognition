import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):

    def __init__(self, root):
        self.samples = []

        # Only keep directories (classes)
        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        for label, cls in enumerate(self.classes):
            folder = os.path.join(root, cls)

            files = [
                f for f in os.listdir(folder)
                if f.endswith(".npy")
            ]

            for f in files:
                path = os.path.join(folder, f)
                self.samples.append((path, label))

        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        clip = np.load(path)

        # Normalize motion (important)
        clip = clip / (np.abs(clip).max() + 1e-6)

        clip = torch.tensor(clip, dtype=torch.float32)
        clip = clip.permute(3, 0, 1, 2)  # (C, T, H, W)

        return clip, label