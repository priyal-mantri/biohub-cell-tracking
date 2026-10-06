"""
Compact 3D U-Net Architecture for Volumetric Cell-Center Heatmap Prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """(Conv3D -> BatchNorm3D -> LeakyReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with MaxPool3D then DoubleConv3D"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling with Trilinear interpolation then DoubleConv3D"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = DoubleConv3D(in_channels + skip_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_z = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x1.size(3)
        diff_x = x2.size(4) - x1.size(4)

        if diff_z != 0 or diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [
                    diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2,
                    diff_z // 2, diff_z - diff_z // 2,
                ]
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Compact 3D U-Net architecture for cell-center response prediction.

    - Input: Tensor of shape (Batch, InChannels=1, Z, Y, X)
    - Output: Tensor of shape (Batch, OutChannels=1, Z, Y, X) in range [0, 1]
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv3D(in_channels, base_features)                # 1 -> 12
        self.down1 = Down3D(base_features, base_features * 2)              # 12 -> 24
        self.down2 = Down3D(base_features * 2, base_features * 4)          # 24 -> 48

        # Decoder
        self.up1 = Up3D(in_channels=base_features * 4, skip_channels=base_features * 2, out_channels=base_features * 2)
        self.up2 = Up3D(in_channels=base_features * 2, skip_channels=base_features, out_channels=base_features)

        # Output head
        self.outc = nn.Conv3d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Heatmap bias initialization: initialize final bias to -4.0 (sigmoid(-4.0) ~ 0.018)
        # Prevents initial loss explosion for sparse keypoint heatmaps
        nn.init.constant_(self.outc.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)        # (B, 12, Z, Y, X)
        x2 = self.down1(x1)     # (B, 24, Z/2, Y/2, X/2)
        x3 = self.down2(x2)     # (B, 48, Z/4, Y/4, X/4)

        x = self.up1(x3, x2)    # (B, 24, Z/2, Y/2, X/2)
        x = self.up2(x, x1)     # (B, 12, Z, Y, X)
        logits = self.outc(x)   # (B, 1, Z, Y, X)
        return self.sigmoid(logits)

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
