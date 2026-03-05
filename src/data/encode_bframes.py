import os
import subprocess
from pathlib import Path

INPUT_ROOT = "raw"
OUTPUT_ROOT = "encoded"

CRF = 23
BFRAMES = 3
GOP = 30
PRESET = "medium"


def reencode_video(input_path, output_path):

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", str(CRF),
        "-bf", str(BFRAMES),
        "-g", str(GOP),
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


def main():

    Path(OUTPUT_ROOT).mkdir(exist_ok=True)

    for class_name in os.listdir(INPUT_ROOT):

        class_input_path = Path(INPUT_ROOT) / class_name

        if not class_input_path.is_dir():
            continue

        class_output_path = Path(OUTPUT_ROOT) / class_name
        class_output_path.mkdir(parents=True, exist_ok=True)

        for video_file in class_input_path.iterdir():

            if video_file.suffix.lower() not in [".avi", ".mp4"]:
                continue

            output_file = class_output_path / (video_file.stem + "_bf.mp4")

            if output_file.exists():
                continue

            print(f"Encoding {class_name}/{video_file.name}")

            success = reencode_video(video_file, output_file)

            # -------- SAFE RAW DELETE --------
            if (
                success
                and output_file.exists()
                and output_file.stat().st_size > 0
            ):
                print(f"Deleting raw file: {video_file}")
                video_file.unlink()
            else:
                print(f"Encoding failed — raw file kept: {video_file}")

    print("Re-encoding complete.")


if __name__ == "__main__":
    main()