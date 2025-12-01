import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()

    def forward(self):
        pass


class IouLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(IouLoss, self).__init__()

    def forward(self):
        pass


class MaskIOULoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskIOULoss, self).__init__()

    def forward(self):
        pass


class CrossEntropyLoss(nn.Module):
    def __init__(self, use_sigmoid=True, loss_weight=1.0):
        super(MaskIOULoss, self).__init__()

    def forward(self):
        pass