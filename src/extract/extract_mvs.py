import av
import numpy as np
import os

os.makedirs("outputs/mvs", exist_ok=True)

container = av.open("samples/test.mp4")
stream = container.streams.video[0]
stream.codec_context.options = {"flags2": "+export_mvs"}

frame_index = 0

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

            np.save(f"outputs/mvs/frame_{frame_index:04d}.npy", arr)

            print(f"Saved B-frame {frame_index} → {arr.shape}")

    frame_index += 1