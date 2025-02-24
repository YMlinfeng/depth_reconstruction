import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        padding,
        dilation,
    ):
        super(_ASPPModule, self).__init__()

        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn(self.atrous_conv(x))
        x = self.relu(x)

        return x


class ASPP(nn.Module):
    def __init__(
        self,
        inplanes,
        mid_channels=None,
        dilations=[1, 6, 12, 18],
        dropout=0.1,
    ):
        super(ASPP, self).__init__()

        if mid_channels is None:
            mid_channels = inplanes // 2

        self.aspp1 = _ASPPModule(
            inplanes, mid_channels, 1, padding=0, dilation=dilations[0]
        )

        self.aspp2 = _ASPPModule(
            inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1]
        )

        self.aspp3 = _ASPPModule(
            inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2]
        )

        self.aspp4 = _ASPPModule(
            inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3]
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # we set the output channel the same as the input
        outplanes = inplanes
        self.conv1 = nn.Conv2d(int(mid_channels * 5), outplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self._init_weight()

    def forward(self, x):
        identity = x.clone()
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return identity + self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class BottleNeckASPP(nn.Module):
    def __init__(
        self,
        inplanes,
        reduction=4,
        dilations=[1, 6, 12, 18],
        dropout=0.1,
    ):
        super(BottleNeckASPP, self).__init__()

        channels = inplanes // reduction
        self.input_conv = nn.Sequential(
            nn.Conv2d(inplanes, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, inplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(
            channels, mid_channels=channels, dropout=dropout, dilations=dilations
        )

    def forward(self, x):
        identity = x
        x = self.input_conv(x)
        x = self.aspp(x)
        x = self.output_conv(x)

        return identity + x
