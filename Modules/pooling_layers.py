import torch.nn as nn

from ._cpools import TopPool, BottomPool, LeftPool, RightPool
from .utils import ConvBnRelu


class CornerPool(nn.Module):
    def __init__(self, channels, poolv, poolh):
        super(CornerPool, self).__init__()
        self.res_conv = nn.Conv2d(channels, channels, (1, 1), bias=False)
        self.res_bn = nn.BatchNorm2d(channels)

        self.v_pre_conv = ConvBnRelu(channels, 64)
        self.h_pre_conv = ConvBnRelu(channels, 64)

        self.conv_add = ConvBnRelu(64, channels, relu=False)
        self.relu_add = nn.ReLU(inplace=True)

        self.post_conv = ConvBnRelu(channels, channels)

        self.vpool = poolv()
        self.hpool = poolh()

    def forward(self, x):
        res = self.res_conv(x)
        res = self.res_bn(res)

        p1_pre = self.v_pre_conv(x)
        pool1 = self.vpool(p1_pre)

        p2_pre = self.h_pre_conv(x)
        pool2 = self.hpool(p2_pre)

        # pool 1 + pool2
        p1_plus_p2 = self.conv_add(pool1 + pool2)

        relu_add = self.relu_add(res + p1_plus_p2)

        return self.post_conv(relu_add)


class TopLeftPool(CornerPool):
    def __init__(self, channels):
        super(TopLeftPool, self).__init__(channels, TopPool, LeftPool)


class BottomRightPool(CornerPool):
    def __init__(self, channels):
        super(BottomRightPool, self).__init__(channels, BottomPool, RightPool)
