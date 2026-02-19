import os
import shutil

src = "outputs/clips"

files = sorted([f for f in os.listdir(src) if f.endswith(".npy")])

classA = os.path.join(src,"classA")
classB = os.path.join(src,"classB")

os.makedirs(classA, exist_ok=True)
os.makedirs(classB, exist_ok=True)

for i,f in enumerate(files):

    if i % 2 == 0:
        shutil.move(os.path.join(src,f), os.path.join(classA,f))
    else:
        shutil.move(os.path.join(src,f), os.path.join(classB,f))

print("Split complete")