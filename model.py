"""CIFAR-sized ResNet-18 and controlled structural variants."""
import torch
from torch import nn


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


def conv3(in_ch, out_ch, stride, depthwise):
    if depthwise:
        return nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                             nn.Conv2d(in_ch, out_ch, 1, bias=False))
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, depthwise=False, se=False):
        super().__init__()
        self.conv1 = conv3(in_ch, out_ch, stride, depthwise)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3(out_ch, out_ch, 1, depthwise)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SE(out_ch) if se else nn.Identity()
        self.shortcut = (nn.Identity() if stride == 1 and in_ch == out_ch else
                         nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return self.relu(self.se(y) + self.shortcut(x))


class ResNet18CIFAR(nn.Module):
    def __init__(self, depthwise=False, se=False, width=32, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        blocks = []
        prev = width
        for stage, count in enumerate((2, 2, 2, 2)):
            channels = width * 2 ** stage
            for i in range(count):
                stride = 2 if stage and i == 0 else 1
                blocks.append(Block(prev, channels, stride, depthwise, se))
                prev = channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(prev, classes)

    def forward(self, x):
        return self.fc(self.pool(self.blocks(self.stem(x))).flatten(1))


def make_model(name, width=32):
    variants = {"baseline": (False, False), "depthwise": (True, False),
                "se": (False, True), "depthwise_se": (True, True)}
    if name not in variants:
        raise ValueError(name)
    depthwise, se = variants[name]
    return ResNet18CIFAR(depthwise, se, width)
