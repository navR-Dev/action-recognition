import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionNetLite(nn.Module):

    def __init__(self, num_classes):
        super(MotionNetLite, self).__init__()

        # Much smaller channels
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        self.pool = nn.MaxPool3d((1, 2, 2))

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x