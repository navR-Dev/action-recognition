import numpy as np
import os

files = os.listdir("outputs/maps")

for f in files[:5]:
    m = np.load("outputs/maps/"+f)
    mag = np.sqrt(m[:,:,0]**2 + m[:,:,1]**2)

    print(f, "mean:", mag.mean(), "max:", mag.max())