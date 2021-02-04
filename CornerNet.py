from .Modules.hourglass import Hourglass
from .Modules.heatmap_utils import  Heatmaps
from torch import nn

class CornerNet(nn.Module):
    def __init__(self):
        super(CornerNet, self).__init__()
        self.hourglass = Hourglass(in_channels=3, out_channels=16)
        self.heatmaps = Heatmaps(in_channels=16, conv_channel=32)

    def forward(self, x, target=None):
        hourg_out = self.hourglass(x)
        if self.training:
            assert target, "give the targets"
            tl_tag_inds, br_tag_inds = target[2:4]  # this is the indices for the tags
            return self.heatmaps(hourg_out, tl_tag_inds, br_tag_inds)
        else:
            return self.heatmaps(hourg_out)
