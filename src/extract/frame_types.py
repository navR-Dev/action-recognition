import subprocess
import json
import sys

video_path = sys.argv[1]

cmd = [
    "ffprobe",
    "-select_streams", "v",
    "-show_frames",
    "-print_format", "json",
    video_path
]

result = subprocess.run(cmd, capture_output=True, text=True)

data = json.loads(result.stdout)

for i, frame in enumerate(data["frames"]):
    if "pict_type" in frame:
        print(f"Frame {i}: {frame['pict_type']}")