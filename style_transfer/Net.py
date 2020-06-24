import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv, self).__init__()
        padding = kernel_size // 2
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, img):
        x = self.pad(img)
        x = self.conv(x)
        return x


class Upsample_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale_factor):
        super(Upsample_conv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        padding = kernel_size // 2
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, img):
        x = self.upsample(img)
        x = self.pad(x)
        x = self.conv(x)
        return x


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.conv1 = Conv(channels, channels, kernel_size=3, stride=1)
        self.instanceNorm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = Conv(channels, channels, kernel_size=3, stride=1)
        self.instanceNorm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, img):
        x = self.conv1(img)
        x = F.relu(self.instanceNorm1(x))
        x = self.conv2(x)
        x = self.instanceNorm2(x)
        x = F.relu(x + img)
        return x


class Style_transfer_net(nn.Module):
    def __init__(self):
        super(Style_transfer_net, self).__init__()

        self.conv1 = Conv(3, 32, kernel_size=9, stride=1)
        self.instanceNorm1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = Conv(32, 64, kernel_size=3, stride=2)
        self.instanceNorm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = Conv(64, 128, kernel_size=3, stride=2)
        self.instanceNorm3 = nn.InstanceNorm2d(128, affine=True)

        self.residual1 = Residual(128)
        self.residual2 = Residual(128)
        self.residual3 = Residual(128)
        self.residual4 = Residual(128)
        self.residual5 = Residual(128)

        self.upsample_conv1 = Upsample_conv(128, 64, kernel_size=3, stride=1, scale_factor=2)
        self.instanceNorm4 = nn.InstanceNorm2d(64, affine=True)
        self.upsample_conv2 = Upsample_conv(64, 32, kernel_size=3, stride=1, scale_factor=2)
        self.instanceNorm5 = nn.InstanceNorm2d(32, affine=True)
        self.conv4 = Conv(32, 3, kernel_size=9, stride=1)

    def forward(self, img):
        x = F.relu(self.instanceNorm1(self.conv1(img)))
        x = F.relu(self.instanceNorm2(self.conv2(x)))
        x = F.relu(self.instanceNorm3(self.conv3(x)))

        # residual blocks
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)

        x = F.relu(self.instanceNorm4(self.upsample_conv1(x)))
        x = F.relu(self.instanceNorm5(self.upsample_conv2(x)))
        # x = torch.tanh(self.conv4(x))
        x = self.conv4(x)
        return x

