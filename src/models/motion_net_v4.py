import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Residual Block ----------------
class ResidualBlock3D(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):

        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out


# ---------------- MotionNet V4 ----------------
class MotionNetV4(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # -------- Initial Feature Extraction --------
        self.conv1 = nn.Conv3d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        # -------- Residual Stage 1 --------
        self.res1 = ResidualBlock3D(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        # -------- Stage 2 --------
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.res2 = ResidualBlock3D(64)
        self.pool2 = nn.MaxPool3d((2, 2, 2))  # temporal pooling added

        # -------- Stage 3 --------
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.res3 = ResidualBlock3D(128)

        # -------- Global Pooling --------
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # -------- Regularization --------
        self.dropout = nn.Dropout(0.3)

        # -------- Classifier --------
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        # Input shape: (B, 2, T, H, W)

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.res1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = self.fc(x)

        return x