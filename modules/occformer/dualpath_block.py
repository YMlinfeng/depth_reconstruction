import torch.nn as nn
import torch.nn.functional as F

# from einops import rearrange
from ..submodules.aspp import BottleNeckASPP
from ..submodules.window_attention import SwinBlock


class DualpathTransformerBlock(nn.Module):
    def __init__(
        self, in_channels, channels, stride=1, coeff_bias=True, aspp_drop=0.1, **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.kwargs = kwargs
        self.shift = (self.kwargs["layer_index"] % 2) == 1

        self.multihead_base_channel = 32
        self.num_heads = int(self.channels / self.multihead_base_channel)

        # build skip connection
        if self.stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(channels),
            )
        else:
            self.downsample = nn.Identity()

        self.input_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

        # shared window attention
        self.bev_encoder = SwinBlock(
            embed_dims=self.channels,
            num_heads=self.num_heads,
            feedforward_channels=self.channels,
            window_size=7,
            drop_path_rate=0.2,
            shift=self.shift,
        )

        # aspp in global path
        self.aspp = BottleNeckASPP(inplanes=self.channels, dropout=aspp_drop)

        # soft weights for fusion
        self.combine_coeff = nn.Conv3d(self.channels, 1, kernel_size=1, bias=coeff_bias)

    def forward(self, x):
        input_identity = x.clone()

        x = self.input_conv(x)
        x = x.squeeze(dim=-1)

        H, W = x.shape[2:]
        win_size = 7
        pad_r = (win_size - W % win_size) % win_size
        pad_b = (win_size - H % win_size) % win_size
        x = F.pad(x, (0, pad_r, 0, pad_b))
        x = self.bev_encoder(x)
        x = x[:, :, :H, :W]

        x = self.aspp(x)
        x = x.unsqueeze(-1)

        return x + self.downsample(input_identity)
