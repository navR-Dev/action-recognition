import torch
from models.motion_net import MotionNet

model = MotionNet(num_classes=5)

x = torch.randn(2,2,8,112,112)

y = model(x)

print(y.shape)