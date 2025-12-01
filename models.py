# resnet50 + fpn + head

import torch
import torch.nn as nn


INF = 1e8

class PolarMaskHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 use_dcn=False,
                 mask_nms=False,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(PolarMaskHead, self).__init__()