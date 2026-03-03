import av
import os

VIDEO_DIR = "samples"   # change if needed


def has_bframe(video_path):
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]

        for frame in container.decode(stream):
            if frame.pict_type == 3:  # B-frame
                return True

        return False

    except Exception as e:
        print(f"ERROR reading {video_path}: {e}")
        return None


def main():
    files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".avi")]

    print(f"Found {len(files)} videos\n")

    for file in files:
        path = os.path.join(VIDEO_DIR, file)
        result = has_bframe(path)

        if result is True:
            print(f"[B] {file}")
        elif result is False:
            print(f"[ ] {file}")
        else:
            print(f"[X] {file}")


if __name__ == "__main__":
    main()