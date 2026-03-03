import numpy as np
import os

MV_ROOT = "outputs/mvs"

for video in os.listdir(MV_ROOT):
    video_path = os.path.join(MV_ROOT, video)
    files = os.listdir(video_path)
    if not files:
        continue

    sample = np.load(os.path.join(video_path, files[0]))
    mags = np.sqrt(sample[:,2]**2 + sample[:,3]**2)

    print(video, "raw mean:", mags.mean(), "raw max:", mags.max())
    break