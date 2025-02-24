# Developed by Junyi Ma
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PoseEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        num_fut_ts=1,
        ):
        super().__init__()
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1
        
        pose_encoder = []

        for _ in range(num_layers - 1):
            pose_encoder.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.pose_encoder = nn.Sequential(*pose_encoder)
    
    def forward(self,x):
        # x: N*2,
        pose_feat = self.pose_encoder(x)
        return pose_feat


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTBlockAttention(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ResidualDiT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        dilation=1,
        norm_cfg=None,
        dit_type='dit',
        nf=64,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        padding_size = [0, 0]
        if dilation!=0:
            padding_size[0] = ((kernel_size[0] - 1) * dilation + 1) // 2
            padding_size[1] = ((kernel_size[1] - 1) * dilation + 1) // 2
        padding_size = tuple(padding_size)

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, dilation=dilation, padding=padding_size)
        self.layers = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True))
        if dit_type == 'dit':
            self.dit = DiTBlock(nf, 4)
        else:
            self.dit = DiTBlockAttention(nf, 4)

    def forward(self, inps):
        inp, pose = inps[0], inps[1]
        b, t, h, w, d, c = pose.shape
        position, condition = torch.chunk(pose, chunks=2, dim=1)

        # add position embedding
        position = rearrange(position, 'b t h w d c -> b (t c d) h w')
        position = F.interpolate(position, size=inp.shape[2:], mode='bilinear')
        inp = inp + position        
        inp = self.layers(inp)

        # reshape x and condition
        condition = rearrange(condition, 'b t h w d c -> b (t c d) h w')
        condition = F.interpolate(condition, size=inp.shape[2:], mode='bilinear')
        _, _, h, w = inp.shape
        inp = rearrange(inp, 'b (t c) h w -> b (t h w) c', b=b, t=t//2, c=c * d)
        condition = rearrange(condition, 'b (t c) h w -> b (t h w) c', b=b, t=t//2, c=c * d)

        # output
        out = self.dit(inp, condition)
        out = rearrange(out, 'b (t h w) c -> b (t c) h w', b=b, t=t//2, h=h, w=w, c=c * d)

        return [out, pose]


class PredictorDiTtime(nn.Module):
    def __init__(
        self,
        n_input_channels=None,
        in_timesteps=None,
        out_timesteps=None,
        norm_cfg=None,
    ):
        super(PredictorDiTtime, self).__init__()
        
        self.predictor = nn.ModuleList()
        self.pose_encoder = nn.ModuleList()
        self.time_encoder = nn.ModuleList()
        for nf in n_input_channels:
            dit_type = 'dit' if nf <= n_input_channels[1] else 'dit_attn'
            self.predictor.append(nn.Sequential(
                ResidualDiT(nf * in_timesteps, nf * in_timesteps, norm_cfg=norm_cfg, nf=nf, dit_type=dit_type),
                ResidualDiT(nf * in_timesteps, nf * in_timesteps, norm_cfg=norm_cfg, nf=nf, dit_type=dit_type),
                ResidualDiT(nf * in_timesteps, nf * out_timesteps, norm_cfg=norm_cfg, nf=nf, dit_type=dit_type),
                ResidualDiT(nf * out_timesteps, nf * out_timesteps, norm_cfg=norm_cfg, nf=nf, dit_type=dit_type),
                ResidualDiT(nf * out_timesteps, nf * out_timesteps, norm_cfg=norm_cfg, nf=nf, dit_type=dit_type),
            ))
            self.pose_encoder.append(PoseEncoder(3, nf // 20))
            self.time_encoder.append(PoseEncoder(1, nf // 20))

    def forward(self, x, time, timestamp=None):
        assert len(x) == len(self.predictor), f'The number of input feature tensors ({len(x)}) must be the same as the number of STPredictor blocks {len(self.predictor)}.'
        
        y = []
        for i in range(len(x)):
            condition = self.time_encoder[i](time)
            if timestamp is not None:
                condition = condition + timestamp[:, None, None, None, None, :]
            y.append(self.predictor[i]([x[i], condition])[0])
                
        return y