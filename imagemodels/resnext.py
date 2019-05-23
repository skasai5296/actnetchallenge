import sys, os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import conv1x1x1, Bottleneck, ResNet
sys.path.append(os.pardir)
from utils.utils import partialclass


def get_inplanes():
    return [128, 256, 512, 1024]


class ResNeXtBottleneck(Bottleneck):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super().__init__(inplanes, planes, stride, downsample)

        mid_planes = cardinality * planes // 32
        self.conv1 = conv1x1x1(inplanes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)


class ResNeXt(ResNet):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 clip_len,
                 imsize,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 cardinality=32,
                 n_classes=739):
        block = partialclass(block, cardinality=cardinality)
        super().__init__(block, layers, block_inplanes=block_inplanes, shortcut_type=shortcut_type, sample_duration=clip_len, sample_size=imsize,
                         num_classes=n_classes)

        self.fc = nn.Linear(cardinality * 32 * block.expansion, n_classes)


def generate_resnext(model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    if model_depth == 50:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 101:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 152:
        model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 200:
        model = ResNeXt(ResNeXtBottleneck, [3, 24, 36, 3], get_inplanes(),
                        **kwargs)

    return model
