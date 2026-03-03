import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionNetV2(nn.Module):

    def __init__(self, num_classes):
        super(MotionNetV2, self).__init__()

        self.conv1 = nn.Conv3d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.pool = nn.MaxPool3d((1, 2, 2))

        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 2, T, H, W)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x