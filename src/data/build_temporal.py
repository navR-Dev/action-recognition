import numpy as np
import os
import cv2
import random
from pathlib import Path

INPUT_DIR = "outputs/maps"
OUTPUT_DIR = "outputs/clips"
T = 8
STRIDE = 2
SIZE = 112
SPLIT_RATIO = 0.8  # 80% train, 20% val

random.seed(42)

# Get all video folders
video_folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]

# Shuffle and split
random.shuffle(video_folders)
split_idx = int(len(video_folders) * SPLIT_RATIO)
train_videos = video_folders[:split_idx]
val_videos = video_folders[split_idx:]

print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

def extract_class_name(folder_name):
    # e.g. v_PlayingGuitar_g14_c02_bf → PlayingGuitar
    return folder_name.split("_")[1]

def save_clips(split, folder):
    class_name = extract_class_name(folder)
    input_path = os.path.join(INPUT_DIR, folder)
    output_path = os.path.join(OUTPUT_DIR, split, class_name)
    os.makedirs(output_path, exist_ok=True)

    files = sorted([f for f in os.listdir(input_path) if f.endswith(".npy")])

    for i in range(0, len(files) - T + 1, STRIDE):
        clip = []

        for j in range(T):
            frame = np.load(os.path.join(input_path, files[i+j]))
            resized = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            clip.append(resized)

        clip = np.stack(clip)  # (T, 112, 112, 2)

        clip_name = f"{folder}_clip_{i:05d}.npy"
        np.save(os.path.join(output_path, clip_name), clip)

        print(f"[{split.upper()}] Saved {clip_name} → {clip.shape}")

# Build all clips
for folder in train_videos:
    save_clips("train", folder)

for folder in val_videos:
    save_clips("val", folder)