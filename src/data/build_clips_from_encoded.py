import os
import av
import numpy as np
import random
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_ROOT = "encoded"
OUTPUT_ROOT = "outputs/clips"

T = 8
STRIDE = 6
SIZE = 112
SPLIT_RATIO = 0.8
MAX_CLIPS_PER_VIDEO = 5
SEED = 42
# ----------------------------------------

random.seed(SEED)


def extract_class_name(video_path):
    # encoded/ClassName/video.mp4
    return Path(video_path).parent.name


def build_motion_map(frame, orig_h, orig_w):
    motion_map = np.zeros((SIZE, SIZE, 2), dtype=np.float16)

    scale_x = SIZE / orig_w
    scale_y = SIZE / orig_h

    for sd in frame.side_data:
        if sd.type.name == "MOTION_VECTORS":
            for mv in sd:
                x = int(mv.src_x * scale_x)
                y = int(mv.src_y * scale_y)

                if 0 <= x < SIZE and 0 <= y < SIZE:
                    motion_map[y, x, 0] = mv.motion_x / 16.0
                    motion_map[y, x, 1] = mv.motion_y / 16.0

    return motion_map


def process_video(video_path, split):

    class_name = extract_class_name(video_path)
    output_dir = Path(OUTPUT_ROOT) / split / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}

    orig_h = stream.codec_context.height
    orig_w = stream.codec_context.width

    motion_maps = []

    # -------- Extract B-frame maps --------
    for frame in container.decode(stream):
        if frame.pict_type == 3:
            motion_map = build_motion_map(frame, orig_h, orig_w)
            motion_maps.append(motion_map)

    if len(motion_maps) < T:
        print(f"[{split.upper()}] {video_path} → 0 clips (too short)")
        return

    # -------- Compute valid clip starts --------
    possible_starts = list(range(0, len(motion_maps) - T + 1, STRIDE))

    if len(possible_starts) > MAX_CLIPS_PER_VIDEO:
        possible_starts = random.sample(
            possible_starts,
            MAX_CLIPS_PER_VIDEO
        )

    clips = []

    for start in possible_starts:
        clip = np.stack(motion_maps[start:start+T])
        clips.append(clip)

    clips = np.stack(clips).astype(np.float16)

    output_file = output_dir / f"{Path(video_path).stem}_clips.npy"
    np.save(output_file, clips)

    print(
        f"[{split.upper()}] {video_path} → "
        f"{clips.shape[0]} clips saved in one file"
    )


def main():

    videos = []

    for root, dirs, files in os.walk(INPUT_ROOT):
        for f in files:
            if f.endswith(".mp4"):
                videos.append(os.path.join(root, f))

    random.shuffle(videos)

    split_idx = int(len(videos) * SPLIT_RATIO)
    train_videos = videos[:split_idx]
    val_videos = videos[split_idx:]

    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos: {len(val_videos)}")

    # -------- Clear previous clips --------
    if Path(OUTPUT_ROOT).exists():
        print("Clearing previous clips...")
        import shutil
        shutil.rmtree(OUTPUT_ROOT)

    for v in train_videos:
        process_video(v, "train")

    for v in val_videos:
        process_video(v, "val")


if __name__ == "__main__":
    main()