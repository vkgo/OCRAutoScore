import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAtt(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction),
                nn.ReLU(),
                nn.Linear(channel//reduction, channel),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CountingDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.trans_layer = nn.Sequential(
            nn.Conv2d(self.in_channel, 512, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(512))
        self.channel_att = ChannelAtt(512, 16)
        self.pred_layer = nn.Sequential(
            nn.Conv2d(512, self.out_channel, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x, mask):
        b, c, h, w = x.size()
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)
        if mask is not None:
            x = x * mask
        x = x.view(b, self.out_channel, -1)
        x1 = torch.sum(x, dim=-1)
        return x1, x.view(b, self.out_channel, h, w)
