import numpy as np
import os

MAP_ROOT = "outputs/maps"

videos = [d for d in os.listdir(MAP_ROOT)
          if os.path.isdir(os.path.join(MAP_ROOT, d))]

for video in videos[:5]:
    video_path = os.path.join(MAP_ROOT, video)
    files = sorted(os.listdir(video_path))

    if not files:
        print(video, "has no maps")
        continue

    sample_file = os.path.join(video_path, files[0])

    m = np.load(sample_file)
    mag = np.sqrt(m[:, :, 0]**2 + m[:, :, 1]**2)

    print(
        video,
        "| shape:", m.shape,
        "| mean:", round(mag.mean(), 4),
        "| max:", round(mag.max(), 4)
    )