import numpy as np
import os

INPUT_DIR = "outputs/mvs"
OUTPUT_DIR = "outputs/maps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HEIGHT = 2160
WIDTH = 3840
BLOCK = 16

for file in sorted(os.listdir(INPUT_DIR)):
    if not file.endswith(".npy"):
        continue

    vectors = np.load(os.path.join(INPUT_DIR, file))

    motion_map = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)

    for x, y, dx, dy in vectors:
        x = int(x)
        y = int(y)

        motion_map[y:y+BLOCK, x:x+BLOCK, 0] = dx
        motion_map[y:y+BLOCK, x:x+BLOCK, 1] = dy

    np.save(os.path.join(OUTPUT_DIR, file), motion_map)

    print("Built map:", file)