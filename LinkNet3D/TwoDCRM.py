import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True) if self.activation else x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.stride = stride
        self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class TwoDCRM(nn.Module):  # you can use BaseBEVBackbone to fit pipeline.
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.down1 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(192, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            ResidualBlock(192, 192)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            ResidualBlock(128, 128)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(192, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            ResidualBlock(192, 192)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            ResidualBlock(256, 256)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            ResidualBlock(512, 512)
        )

        self.num_bev_features = 512

    def forward(self, data_dict):
        """
        Forward pass of the TwoDCRM model.

        Args:
            data_dict (dict): Dictionary containing 'spatial_features'.

        Returns:
            dict: Updated dictionary with 'spatial_features_2d'.
        """
        x = data_dict['spatial_features']
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        data_dict['spatial_features_2d'] = x
        return data_dict

