import torch
from torch import nn


def focal_loss(pred_mp: torch.Tensor, gt_mp: torch.Tensor):
    # gt map must be created with the gaussian thing
    pos_inds = gt_mp.eq(1)
    neg_inds = gt_mp.lt(1)

    pos_preds = pred_mp[pos_inds]
    neg_preds = pred_mp[neg_inds]

    gt_negs = gt_mp[neg_inds]

    pos_loss = torch.log(pos_preds) * torch.pow(1 - pos_preds, 2)
    neg_loss = torch.log(1 - neg_preds) * torch.pow(neg_preds, 2) * torch.pow(1 - gt_negs, 4)

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_preds.nelement() == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss


def push_pull_loss(tag_tl: torch.Tensor, tag_br: torch.Tensor, tag_mask: torch.Tensor):
    # the ind maps will also have a fixed number of maps. so we need the mask
    num = tag_mask.sum(dim=1, keepdim=True).float()
    tag_tl = tag_tl.squeeze()
    tag_br = tag_br.squeeze()

    tag_mean = (tag_tl + tag_br) / 2

    tag_tl = torch.pow(tag_tl - tag_mean, 2) / (num + 1e-4)
    tag_tl = tag_tl[tag_mask].sum()
    tag_br = torch.pow(tag_br - tag_mean, 2) / (num + 1e-4)
    tag_br = tag_br[tag_mask].sum()
    pull = tag_tl + tag_br

    tag_mask = tag_mask.unsqueeze(1) + tag_mask.unsqueeze(2)
    tag_mask = tag_mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[tag_mask]
    push = dist.sum()
    return pull, push


def regression_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class TotalLoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1):
        super(TotalLoss, self).__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight

    def forward(self, pred, target):
        tl_heatmp, br_heatmp = pred[:2]
        tl_tags, br_tags = pred[2:4]
        tl_regs, br_regs = pred[4:6]

        targ_tl_heat, targ_br_heat = target[:2]
        tl_tag_inds, br_tag_inds = target[2:4]
        targ_tl_reg, targ_br_reg = target[4:6]
        tag_masks = target[7]

        heatmap_focal = focal_loss(tl_heatmp, targ_tl_heat) + (br_heatmp + targ_br_heat)
        embed_push, embed_pull = push_pull_loss(tl_tags, br_tags, tag_masks)
        regress_loss = regression_loss(tl_regs, targ_tl_reg, tag_masks) \
                       + regression_loss(br_regs, targ_br_reg, tag_masks)

        final_loss = heatmap_focal + self.push_weight * embed_push + \
                     self.pull_weight * embed_pull * self.regr_weight * regress_loss

        return final_loss.unsqueeze(0)
