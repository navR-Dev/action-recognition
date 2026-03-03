import os
import subprocess
from pathlib import Path

INPUT_ROOT = "raw"
OUTPUT_ROOT = "encoded"

CRF = 23
BFRAMES = 3
GOP = 30
PRESET = "medium"

def get_class_name(filename):
    # UCF format: v_ClassName_gXX_cYY.avi
    name = filename.split("_")
    return name[1]

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

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    Path(OUTPUT_ROOT).mkdir(exist_ok=True)

    for file in os.listdir(INPUT_ROOT):

        if not file.lower().endswith((".avi", ".mp4")):
            continue

        class_name = get_class_name(file)

        class_output = Path(OUTPUT_ROOT) / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        input_path = Path(INPUT_ROOT) / file
        output_file = class_output / (Path(file).stem + "_bf.mp4")

        if output_file.exists():
            continue

        print(f"Encoding {file} → {class_name}/")
        reencode_video(input_path, output_file)

    print("Re-encoding complete.")

if __name__ == "__main__":
    main()