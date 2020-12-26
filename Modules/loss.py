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
