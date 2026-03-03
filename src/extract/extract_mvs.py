import av
import numpy as np
import os

INPUT_DIR = "encoded"
OUTPUT_ROOT = "outputs/mvs"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def extract_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(OUTPUT_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing: {video_name}")

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}

    frame_index = 0
    saved_count = 0

    for frame in container.decode(stream):

        if frame.pict_type == 3:  # B-frame

            vectors = []

            for sd in frame.side_data:
                if sd.type.name == "MOTION_VECTORS":
                    for mv in sd:
                        scale = 16.0
                        vectors.append([
                            mv.src_x,
                            mv.src_y,
                            mv.motion_x / scale,
                            mv.motion_y / scale
                        ])

            if vectors:
                arr = np.array(vectors, dtype=np.float32)
                np.save(
                    os.path.join(output_dir, f"frame_{frame_index:04d}.npy"),
                    arr
                )
                saved_count += 1

        frame_index += 1

    print(f"Saved {saved_count} B-frames")


def main():
    videos = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith((".mp4", ".avi")):
                videos.append(os.path.join(root, f))

    print(f"Found {len(videos)} videos")

    for video in videos:
        try:
            extract_video(video)
        except Exception as e:
            print(f"Error processing {video}: {e}")

if __name__ == "__main__":
    main()