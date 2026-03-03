import numpy as np
import os

INPUT_ROOT = "outputs/mvs"
OUTPUT_ROOT = "outputs/maps"

TARGET_H = 224
TARGET_W = 224
BLOCK = 16  # macroblock size in original

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def build_map(vectors, orig_h, orig_w):
    motion_map = np.zeros((TARGET_H, TARGET_W, 2), dtype=np.float32)

    if len(vectors) == 0:
        return motion_map

    scale_x = TARGET_W / orig_w
    scale_y = TARGET_H / orig_h

    for x, y, dx, dy in vectors:
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)

        block_w = max(1, int(BLOCK * scale_x))
        block_h = max(1, int(BLOCK * scale_y))

        motion_map[
            y_scaled:y_scaled+block_h,
            x_scaled:x_scaled+block_w,
            0
        ] = dx

        motion_map[
            y_scaled:y_scaled+block_h,
            x_scaled:x_scaled+block_w,
            1
        ] = dy

    return motion_map


def process_video(video_folder):
    video_name = os.path.basename(video_folder)
    output_dir = os.path.join(OUTPUT_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(video_folder))

    for f in files:
        vectors = np.load(os.path.join(video_folder, f))

        # Assume original resolution known (use one fixed resolution for now)
        orig_h = 2160
        orig_w = 3840

        motion_map = build_map(vectors, orig_h, orig_w)

        np.save(
            os.path.join(output_dir, f.replace(".npy", "_map.npy")),
            motion_map
        )

    print(f"Processed {video_name}")


def main():
    videos = [
        os.path.join(INPUT_ROOT, d)
        for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    print(f"Found {len(videos)} videos")

    for video in videos:
        process_video(video)


if __name__ == "__main__":
    main()