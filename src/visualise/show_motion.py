import numpy as np
import cv2

path = "outputs/maps/frame_0009.npy"   # change file to test

flow = np.load(path)

dx = flow[:, :, 0]
dy = flow[:, :, 1]

mag, ang = cv2.cartToPolar(dx, dy)

hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 1] = 255
mag = mag * 15   # amplify motion
hsv[...,2] = np.clip(mag, 0, 255)

rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("motion", rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()