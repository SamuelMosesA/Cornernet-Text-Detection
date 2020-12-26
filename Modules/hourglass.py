import torch.nn as nn

from Modules.utils import ConvBnRelu


class NormalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), padding=(2, 2), stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.bn1(conv)
        return self.relu1(conv)


class rec_hourglass(nn.Module):
    def __init__(self, module_channels, div_factor):
        super(rec_hourglass, self).__init__()

        current_channel = module_channels[0]
        next_channel = module_channels[1]
        m_d_factor = div_factor[1] // div_factor[0]

        next_channels = module_channels[1:]
        next_div_factor = div_factor[1:]

        self.skip_conv = nn.Conv2d(current_channel, current_channel, kernel_size=(1, 1), stride=1)
        self.down_conv = ConvBnRelu(current_channel, next_channel, k=m_d_factor, pad=1, stride=m_d_factor)
        # self.down_sample = nn.MaxPool2d(kernel_size=m_d_factor, stride=m_d_factor, return_indices=True)

        self.middle = rec_hourglass(next_channels, next_div_factor) if len(next_channels) > 1 \
            else NormalConv(next_channel, next_channel)

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=m_d_factor)
        self.up_conv = ConvBnRelu(next_channel, current_channel)

    def forward(self, x):
        skip_c = self.skip_conv(x)
        conv_l = self.down_conv(x)
        middle = self.middle(conv_l)
        up_sample = self.up_sample(middle)
        up_sample = self.up_conv(up_sample)
        return skip_c + up_sample


class Hourglass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Hourglass, self).__init__()
        module_channels = [16, 64, 64, 128]
        module_div = [4, 8, 16, 32]

        self.pre = nn.Sequential(nn.Conv2d(in_channels, 4, kernel_size=3, padding=1),
                                 ConvBnRelu(4, 8),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 ConvBnRelu(8, 16),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 )
        self.hg = rec_hourglass(module_channels, module_div)
        self.post = nn.Conv2d(4, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.pre(x)
        hg = self.hg(c1)
        return self.post(hg)
