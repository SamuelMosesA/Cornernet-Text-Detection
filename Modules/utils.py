from typing import Tuple

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, pad=-1, stride=1, bias=False, relu=True):
        super().__init__()
        if pad == -1:
            pad = (k - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(k, k), padding=(pad, pad),
                               stride=(stride, stride), bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True) if relu else nn.Sequential()

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.bn1(conv)
        return self.relu1(conv)


class ConvWithK1Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, k=3, pad=-1, stride=1, bias=False, relu=True):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_channels, mid_channels, k=3),
                                  nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), bias=bias))

    def forward(self, x):
        return self.conv(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.reluf = nn.ReLU(inplace=True)

    def forward(self, x):
        c_1 = self.conv_1(x)
        c_1 = self.bn_1(c_1)
        c_1 = self.relu1(c_1)

        c_2 = self.conv_2(c_1)
        c_2 = self.bn_2(c_2)

        res = self.residual(x)
        return self.reluf(res + c_2)


def permute_gather_ind_values(inp_map: torch.Tensor, indices: torch.Tensor):
    inp_map = inp_map.permute(0, 2, 3, 1).contiguous()
    batch, h, w, c = inp_map.size()
    inp_map = inp_map.view(batch, -1, c)

    indices = indices.unsqueeze(2)
    gathered = inp_map.gather(1, indices)

    return gathered


def gather_ind(inp_map: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    dim = inp_map.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

    return inp_map.gather(1, ind)


def conv_nms(inp: torch.Tensor) -> torch.Tensor:
    kernel = 3
    pad = (kernel - 1) // 2

    pooled = nn.functional.max_pool2d(inp, kernel_size=(kernel, kernel), padding=pad, stride=1)
    ind_keep = (inp == pooled).float()
    return inp * ind_keep


def top_k(inp: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, c, height, width = inp.size()
    scores, inds = torch.topk(inp.view(batch, -1), k)
    ys = (inds / width).int().float()
    xs = (inds % width).int().float()

    return scores, inds, xs, ys
