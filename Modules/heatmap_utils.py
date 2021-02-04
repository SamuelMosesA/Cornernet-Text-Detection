from typing import List

import torch
from torch import nn

from Modules.pooling_layers import TopLeftPool, BottomRightPool
from Modules.utils import ConvBnRelu, ConvWithK1Conv, permute_gather_ind_values, gather_ind, conv_nms, top_k


class heatmaps(nn.Module):
    def __init__(self, in_channels, conv_channel):
        super().__init__()
        self.pre_heat_conv = ConvBnRelu(in_channels, conv_channel, k=3)
        self.tl_pool = TopLeftPool(conv_channel)
        self.br_pool = BottomRightPool(conv_channel)

        self.tl_heats = ConvWithK1Conv(conv_channel, in_channels, 1)
        self.br_heats = ConvWithK1Conv(conv_channel, in_channels, 1)

        self.tl_tag = ConvWithK1Conv(conv_channel, in_channels, 1)
        self.br_tag = ConvWithK1Conv(conv_channel, in_channels, 1)

        self.tl_regr = ConvWithK1Conv(conv_channel, in_channels, 2)
        self.br_regr = ConvWithK1Conv(conv_channel, in_channels, 2)

    @staticmethod
    def _train(heats: List[torch.Tensor], *args):
        tl_inds: torch.Tensor = args[1]
        br_inds: torch.Tensor = args[2]

        tl_heatmp, br_heatmp = heats[0:2]
        tl_tagmp, br_tagmp = heats[2:4]
        tl_regmp, br_regmp = heats[4:6]

        tl_tags, br_tags = permute_gather_ind_values(tl_tagmp, tl_inds), permute_gather_ind_values(br_tagmp, br_inds)
        tl_regs, br_regs = permute_gather_ind_values(tl_regmp, tl_inds), permute_gather_ind_values(br_regmp, br_inds)

        return [tl_heatmp, br_heatmp, tl_tags, br_tags, tl_regs, br_regs]

    @staticmethod
    def _eval(heats, *args):
        tl_heatmp, br_heatmp = heats[0:2]
        tl_tagmp, br_tagmp = heats[2:4]
        tl_regmp, br_regmp = heats[4:6]

        batch, c, height, width = tl_heatmp.size()

        tl_heatmp, br_heatmp = torch.sigmoid(tl_heatmp), torch.sigmoid(br_heatmp)

        K = 400
        ae_threshold = 0.5
        num_dets = 1000

        tl_heatmp, br_heatmp = conv_nms(tl_heatmp), conv_nms(br_heatmp)
        tl_scores, tl_inds, tl_xs, tl_ys = top_k(tl_heatmp, K)
        br_scores, br_inds, br_xs, br_ys = top_k(br_heatmp, K)

        # this view diffent dimensions is to make all possible combinations of tl and br
        # they are like the column and row labels
        tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
        tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
        br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
        br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)

        tl_regs, br_regs = permute_gather_ind_values(tl_regmp, tl_inds), permute_gather_ind_values(br_regmp, br_inds)
        tl_regs = tl_regs.view(batch, K, 1, 2)
        br_regs = br_regs.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regs[..., 0]
        tl_ys = tl_ys + tl_regs[..., 1]
        br_xs = br_xs + br_regs[..., 0]
        br_ys = br_ys + br_regs[..., 1]

        tl_tags = permute_gather_ind_values(tl_tagmp, tl_inds)
        tl_tags = tl_tags.view(batch, K, 1)
        br_tags = permute_gather_ind_values(br_tagmp, br_inds)
        br_tags = br_tags.view(batch, 1, K)
        dists = torch.abs(tl_tags - br_tags)

        tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
        br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
        scores = (tl_scores + br_scores) / 2

        # rejecting not possible boxes
        dists_inds = (dists > ae_threshold)
        width_inds = (br_xs < tl_xs)
        height_inds = (br_ys < tl_ys)

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

        scores[dists_inds] = -1
        scores[width_inds] = -1
        scores[height_inds] = -1

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)
        bboxes = bboxes.view(batch, -1, 4)
        bboxes = gather_ind(bboxes, inds)

        tl_scores = tl_scores.contiguous().view(batch, -1, 1)
        tl_scores = gather_ind(tl_scores, inds).float()
        br_scores = br_scores.contiguous().view(batch, -1, 1)
        br_scores = gather_ind(br_scores, inds).float()

        detections = torch.cat([bboxes, scores, tl_scores, br_scores], dim=2)

        return detections

    def forward(self, *args):
        hourg_out = args[0]
        pre = self.pre_heat_conv(hourg_out)
        tl = self.tl_pool(pre)
        br = self.br_pool(pre)

        tl_heatmp, br_heatmp = self.tl_heats(tl), self.br_heats(br)
        tl_tagmp, br_tagmp = self.tl_tag(tl), self.br_tag(br)
        tl_regmp, br_regmp = self.tl_regr(tl), self.br_regr(br)

        heats = [tl_heatmp, br_heatmp, tl_tagmp, br_tagmp, tl_regmp, br_regmp]

        if self.training:
            assert len(args) == 3, "3 arguments expected"
            return self._train(heats, *args)
        else:
            return self._eval(heats, *args)
