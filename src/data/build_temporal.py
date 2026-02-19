import numpy as np
import os
import cv2

INPUT_DIR = "outputs/maps"
OUTPUT_DIR = "outputs/clips"

T = 8
SIZE = 112   # target resolution

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")])

for i in range(len(files) - T + 1):

    clip = []

    for j in range(T):
        frame = np.load(os.path.join(INPUT_DIR, files[i+j]))

        # resize motion map
        resized = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

        clip.append(resized)

    clip = np.stack(clip)   # (T,112,112,2)

    name = f"clip_{i:05d}.npy"
    np.save(os.path.join(OUTPUT_DIR, name), clip)

    print("Saved", name, clip.shape)