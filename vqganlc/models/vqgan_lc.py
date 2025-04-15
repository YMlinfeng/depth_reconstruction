import torch  # 导入 PyTorch 库，用于张量操作和神经网络构建（Tensor类型：torch.Tensor）
import torch.nn.functional as F  # 导入常用的神经网络函数，如激活函数、损失函数（例如 F.relu, F.mse_loss 等）
import importlib  # 导入 importlib 模块，用于动态导入模块
from einops import rearrange  # 导入 rearrange 函数，可灵活改变张量的形状，例如将 (B, C, H, W) 变为 (B, H, W, C)
from torch.nn import Embedding  # 导入 Embedding 层，用于构造词嵌入表（例如码本）
from vqganlc.models.discriminator import NLayerDiscriminator, weights_init  # 导入鉴别器模型和权重初始化函数
from vqganlc.models.lpips import LPIPS  # 导入 LPIPS 模块，用于感知相似度计算（衡量高层特征差异）
from vqganlc.models.encoder_decoder import Encoder, Decoder, Decoder_Cross  # 导入编码器/解码器模块
""" adopted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py """
# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import numpy as np
import math
from vqvae.utils import shift_dim , view_range
import torch.nn.functional as F
from copy import deepcopy
from vqvae.attention import *
from vqvae.quantizer import VectorQuantizer
from einops import rearrange

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

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
    
    def forward(self, x, shape):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        diffY = shape[0] - x.size()[2]
        diffX = shape[1] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
    
    def forward(self, x):
        if self.with_conv:
            #pad = (0, 1, 0, 1, 0, 1)
            #x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class VAERes3DVoxel(nn.Module):
    def __init__(
            self,
            inp_channels=4,
            out_channels=4,
            mid_channels=256,
            z_channels=8,
            vqvae_cfg=None,
            ):
        super().__init__()
        self.pre_vq_conv = SamePadConv3d(mid_channels, z_channels, 1)
        self.encoder_gpt = Encoder(mid_channels, 4, (4, 4, 4), in_channels=mid_channels)  # args.n_hiddens, args.n_res_layers, args.downsample
        self.post_vq_conv = SamePadConv3d(z_channels, mid_channels, 1)
        self.decoder_gpt = Decoder(mid_channels, 4, (4, 4, 4), out_channel=mid_channels)
        self.embedder = nn.Linear(inp_channels, mid_channels)
        self.embedder_t = nn.Linear(mid_channels, out_channels)
        self.vqvae = VectorQuantizer(n_e=mid_channels, e_dim=z_channels, beta=1., z_channels=z_channels, use_voxel=True)
    
    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        bs, C, H, W, D = x.shape
        x = rearrange(x, 'b c h w d -> b h w d c')
        x = self.embedder(x)
        x = rearrange(x, 'b h w d c -> b c h w d')
        x = self.pre_vq_conv(self.encoder_gpt(x))
        return x
             
    def forward_decoder(self, z, input_shape):
        x = self.decoder_gpt(self.post_vq_conv(z))
        bs, C, H, W, D = input_shape
        similarity = self.embedder_t(x.permute(0, 2, 3, 4, 1))
        similarity = rearrange(similarity, 'b h w d c -> b c h w d')
        return similarity

    def forward(self, x, **kwargs):
        output_dict = {}

        z = self.forward_encoder(x)
        z_sampled, loss, info = self.vqvae(z, is_voxel=False)
        output_dict.update({'embed_loss': loss})
        
        mid = z_sampled
        logits = self.forward_decoder(z_sampled, x.shape)
        
        output_dict.update({'logits': logits})
        output_dict.update({'mid': mid})
        return output_dict
        
    def generate(self, z, input_shape):
        logits = self.forward_decoder(z, input_shape)
        return {'logits': logits}


class VAERes2D(nn.Module):
    def __init__(
            self,
            inp_channels=80,
            out_channels=320,
            vqvae_cfg=None,
            ):
        super().__init__()
        self.pre_vq_conv = SamePadConv3d(640, 320, 1)
        self.encoder_gpt = Encoder(640, 4, (4, 4, 1), in_channels=320)  # args.n_hiddens, args.n_res_layers, args.downsample
        self.post_vq_conv = SamePadConv3d(320, 640, 1)
        self.decoder_gpt = Decoder(640, 4, (4, 4, 1), out_channel=320)
        self.embedder = nn.Linear(inp_channels, out_channels)
        self.embedder_t = nn.Linear(out_channels, inp_channels)
        self.vqvae = VectorQuantizer(n_e=640, e_dim=320, beta=1., z_channels=320, use_voxel=True)
    
    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        bs, C, H, W, D = x.shape
        x = rearrange(x, 'b c h w d -> b h w (c d)')
        x = self.embedder(x)
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        x = self.pre_vq_conv(self.encoder_gpt(x))
        return x
             
    def forward_decoder(self, z, input_shape):
        x = self.decoder_gpt(self.post_vq_conv(z))
        bs, C, H, W, D = input_shape
        similarity = self.embedder_t(x.permute(0, 2, 3, 4, 1))
        similarity = rearrange(similarity, 'b h w dd (c d) -> b c h w (dd d)', dd=1, d=D)
        return similarity

    def forward(self, x, **kwargs):
        output_dict = {}

        z = self.forward_encoder(x)
        z_sampled, loss, info = self.vqvae(z, is_voxel=False)
        output_dict.update({'embed_loss': loss})
        
        mid = z_sampled
        logits = self.forward_decoder(z_sampled, x.shape)
        
        output_dict.update({'logits': logits})
        output_dict.update({'mid': mid})
        return output_dict
        
    def generate(self, z, input_shape):
        logits = self.forward_decoder(z, input_shape)
        return {'logits': logits}


class VAERes3D(nn.Module):
    def __init__(
            self,
            inp_channels=4,
            out_channels=128,
            vqvae_cfg=None,
            ):
        super().__init__()
        self.pre_vq_conv = SamePadConv3d(512, 128, 1)
        self.encoder_gpt = Encoder(512, 4, (4, 4, 4))  # args.n_hiddens, args.n_res_layers, args.downsample
        self.post_vq_conv = SamePadConv3d(128, 512, 1)
        self.decoder_gpt = Decoder(512, 4, (4, 4, 4))
        self.embedder = nn.Linear(inp_channels, out_channels)
        self.embedder_t = nn.Linear(out_channels, inp_channels)
        self.vqvae = VectorQuantizer(n_e=512, e_dim=128, beta=1., z_channels=128, use_voxel=True)
    
    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        bs, C, H, W, D = x.shape
        x = self.embedder(x.permute(0, 2, 3, 4, 1))
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pre_vq_conv(self.encoder_gpt(x))
        return x 
             
    def forward_decoder(self, z, input_shape):
        x = self.decoder_gpt(self.post_vq_conv(z))
        bs, C, H, W, D = input_shape
        similarity = self.embedder_t(x.permute(0, 2, 3, 4, 1))
        return similarity.permute(0, 4, 1, 2, 3)

    def forward(self, x, **kwargs):
        output_dict = {}

        z = self.forward_encoder(x)
        z_sampled, loss, info = self.vqvae(z, is_voxel=False)
        output_dict.update({'embed_loss': loss})
        
        mid = z_sampled
        logits = self.forward_decoder(z_sampled, x.shape)
        
        output_dict.update({'logits': logits})
        output_dict.update({'mid': mid})
        return output_dict
        
    def generate(self, z, input_shape):
        logits = self.forward_decoder(z, input_shape)
        return {'logits': logits}


class Encoder2D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Enc has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # x: bs, F, H, W, D
        shapes = []
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                shapes.append(h.shape[-2:])
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        #
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes


class Decoder2D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align with encoder
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, shapes):
        # z: bs*F, C, H, W
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        print("ln_decoder_666",z.shape)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align encoder
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, shapes.pop())

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2)
        )

    def forward(self, x):
        return x + self.block(x)


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer,
                 causal, attn_type, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == 'full':
            self.attn = FullAttention(shape, causal, **attn_kwargs)
        elif attn_type == 'axial':
            assert not causal, 'causal axial attention is not supported'
            self.attn = AxialAttention(len(shape), **attn_kwargs)
        elif attn_type == 'sparse':
            self.attn = SparseAttention(shape, n_head, causal, **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None

        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                                    v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None), *[slice(i, i+ 1) for i in decode_idx])
                self.cache['k'][idx] = k
                self.cache['v'][idx] = v
            k, v = self.cache['k'], self.cache['v']

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a) # (b x seq_len x embd_dim)

        return a


class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2 # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample, out_channel=128):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = out_channel if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4,
                                           stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample, in_channels=128):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = in_channels if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h













# ----------------------------------------------------------------------------
# 模块说明：通过字符串获取对象实例（从动态加载的模块中返回类/函数）
# 主要用于根据配置文件中的字符串路径动态实例化模块
# ----------------------------------------------------------------------------
def get_obj_from_str(string, reload=False):
    # 参数 string: 字符串形式的模块路径和类名，例如 "module.submodule.ClassName"
    module, cls = string.rsplit(".", 1)  # 将字符串按最后一个点分隔，module（模块名），cls（类名）
    if reload:
        module_imp = importlib.import_module(module)  # 导入模块
        importlib.reload(module_imp)  # 重新加载模块，确保得到最新状态
    # 返回模块中对应的属性（一般是类），用于后续用关键字参数实例化对象
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    """
    宏观目的：
    根据传入的配置字典 config【包含 'target' 和 可选 'params'】实例化对应的对象。
    config 示例：{'target': 'module.submodule.ClassName', 'params': {'param1': value1, ...}}
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")  # 键不存在则报错
    # 使用 get_obj_from_str 获得目标类，并展开参数字典(**params)进行实例化。
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# ----------------------------------------------------------------------------
# 模块说明：定义 VQGAN-LC 模型类
#
# 该模型主要包含：
# 1. 编码器（Encoder）：接受输入图像并转换成潜在空间的表示
# 2. 量化器（Quantizer）：将连续的潜在表示映射到离散的码本嵌入上
# 3. 解码器（Decoder）：将离散码转换回图像
# 4. 鉴别器（Discriminator）：用于计算对抗损失，提高图像生成质量
#
# 同时包含了 EMA（指数移动平均）机制的实现和自适应权重计算，用于平衡
# 重构损失、量化损失、感知损失以及对抗损失。
# ----------------------------------------------------------------------------
class VQModel(torch.nn.Module):
    def __init__(self,
                args,  # 参数对象，包含模型训练和结构的配置属性（例如 stage、embed_dim、n_vision_words、quantizer_type 等）
                # embed_dim,  # 嵌入维度（整数），例如256、512，决定了潜在空间和码本的维度
                ddconfig={},  # 字典格式的配置，指定 Encoder/Decoder 的网络结构（例如 z_channels、分辨率、层数等）
                ckpt_path=None,  # 可选的预训练模型文件路径，用于加载已有的权重
                ignore_keys=[],  # 加载权重时需要忽略的键列表
                image_key="image",  # 输入数据字典中图像对应的键名
                remap=None,  # 可选参数，若需要重新映射输出编码索引则会用到
                sane_index_shape=False,  # 标志，若 True，则量化器返回的索引形状为 (batch, height, width)
                inp_channels=3,      # 输入图像的通道数，默认RGB图像（3通道）--80
                out_channels=3,      # 输出图像通道数，通常与输入图像通道数一致 --80
                mid_channels=256,    # Encoder的输出通道，用于投射到更高维特征空间 -- 1024
                z_channels=4,        # 潜变量 z 的通道数（通常较小，用于压缩表示）-- 256
                vqvae_cfg=None,      # VQ-VAE的配置（当前版本中未使用该变量，可扩展）
                ):
        super().__init__()

        self.pre_vq_conv = SamePadConv3d(mid_channels, z_channels, 1)
        
        # 参数：输入通道 mid_channels，残差层数24，下采样率为 (2,2,1) #通道维度从inp到mid
        self.encoder_gpt = Encoder(mid_channels, 24, (2, 2, 1), in_channels=mid_channels)
        
        # post_vq_conv: 用于将经过 VQ-VAE 量化后的 z，再映射回中间表示空间
        self.post_vq_conv = SamePadConv3d(z_channels, mid_channels, 1)
        
        # decoder_gpt: 解码器模块，将中间表示还原，参数与 encoder 对应
        self.decoder_gpt = Decoder(mid_channels, 24, (2, 2, 1), out_channel=mid_channels)
        
        self.embedder = nn.Linear(inp_channels, mid_channels)
        
        # embedder_t: 将中间表示转换为最终输出图像通道（out_channels），类似于最后的线性投射
        self.embedder_t = nn.Linear(mid_channels, out_channels)
        

        # self.image_key = image_key  # 保存输入数据的图像键
        self.args = args  # 保存参数对象
        
        # self.encoder = Encoder(**ddconfig)  # 使用 ddconfig 初始化编码器；输入图像 shape: (B, 3, H, W)
        # self.decoder = Decoder(**ddconfig)  # 使用 ddconfig 初始化解码器；输出图像 shape 应与输入匹配
        # 初始化鉴别器，输入为 3 通道图像，采用两层卷积（n_layers=2），不使用 actnorm
        # self.discriminator = NLayerDiscriminator(input_nc=3,
        #                                          n_layers=2,
        #                                          use_actnorm=False,
        #                                          ndf=64
        #                                         ).apply(weights_init)  # 对鉴别器参数进行权重初始化
        
        embed_dim = z_channels  # 从参数中重新获得嵌入向量维度
        # self.perceptual_loss = LPIPS().eval()  # 初始化 LPIPS 模块，用于计算感知损失，设置为 eval() 模式避免训练
        # self.perceptual_weight = args.rate_p  # 感知损失的权重因子，用于 loss 加权
        self.quantize_type = args.quantizer_type  # 量化器类型，例如 "ema"（指数移动平均）等

        # print("****Using Quantizer: %s" % (args.quantizer_type))
        self.criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失，一般用于分类任务，此处可能用于码本预测
        
        # if ckpt_path is not None:
        #     self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)  # 如果有预训练权重，则加载
        self.image_key = image_key
        # if colorize_nlabels is not None:
        #     # 若提供 colorize_nlabels，则生成一个随机的颜色映射矩阵，形状为 (3, colorize_nlabels, 1, 1)
        #     assert type(colorize_nlabels)==int
        #     self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        # if monitor is not None:
        #     self.monitor = monitor  # 保存监控变量
        
        # 初始化码本（词嵌入），codebook_dim 即码本向量的维度，初始与 embed_dim 相同
        codebook_dim = embed_dim
        if args.tuning_codebook == -1:  # Random：随机初始化且允许调优 #! 实现了这个逻辑
            print("****Using Tuned Random Codebook****")
            print("Word Number:%d" % (args.n_vision_words))
            print("Feature Dim:%d" % (embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, embed_dim)  # 构造一个大小为 (n_vision_words, embed_dim) 的嵌入表
            # 均匀初始化，数值范围为 (-1/n_vision_words, 1/n_vision_words)
            self.tok_embeddings.weight.data.uniform_(-1.0 / args.n_vision_words, 1.0 / args.n_vision_words)
            self.tok_embeddings.weight.requires_grad = True  # 允许梯度更新
            
        elif args.tuning_codebook == -2:  # Random Fix：随机初始化但固定，不调优 #!未测试
            print("****Using Fix Random Codebook****")
            print("Word Number:%d" % (args.n_vision_words))
            print("Feature Dim:%d" % (embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, embed_dim)
            self.tok_embeddings.weight.data.uniform_(-1.0 / args.n_vision_words, 1.0 / args.n_vision_words)
            self.tok_embeddings.weight.requires_grad = False  # 固定，不更新梯度

        elif args.tuning_codebook == 0:  # Fix Initialized Codebook：加载预训练权重并固定 #! 未测试
            print("****Using Fix Initialized Codebook****")
            checkpoint = torch.load(args.local_embedding_path, map_location="cpu")
            args.n_vision_words = checkpoint.shape[0]  # 更新词数量为 checkpoint 的行数
            codebook_dim = checkpoint.shape[1]  # 更新码本向量维度
            print("Word Number:%d" % (args.n_vision_words))
            print("Feature Dim:%d" % (embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, checkpoint.shape[1])
            self.tok_embeddings.weight.data = checkpoint  # 加载预训练 embedding
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
            self.tok_embeddings.weight.requires_grad = False  # 固定权重

        elif args.tuning_codebook == 1:  # Tuning Initialized Codebook：加载预训练权重但允许调优 #! 未测试
            print("****Tuning Initialized Codebook****")
            checkpoint = torch.load(args.local_embedding_path, map_location="cpu")
            args.n_vision_words = checkpoint.shape[0]
            codebook_dim = checkpoint.shape[1]
            print("Word Number:%d" % (args.n_vision_words))
            print("Feature Dim:%d" % (embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, checkpoint.shape[1])
            self.tok_embeddings.weight.data = checkpoint
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
            self.tok_embeddings.weight.requires_grad = True  # 允许调优

        self.e_dim = embed_dim  # 保存嵌入维度，后续用于重构 z 张量
        self.remap = remap  # 保存 remap 信息（若需要对量化后的索引进行修改）
        # self.sane_index_shape = sane_index_shape  # 标志是否返回 (B, H, W) 形状的索引
        self.sane_index_shape = sane_index_shape  # 标志是否返回 (B, D, H, W) 形状的索引
        
        # # 定义量化前的卷积层：将编码器输出的通道数转换到嵌入维度 embed_dim
        # # 输入: (B, z_channels, H, W) ； 输出: (B, embed_dim, H, W)
        # self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        # # 定义反量化卷积层，将嵌入空间映射回编码器原始的 z_channels
        # # 输入: (B, embed_dim, H, W) ； 输出: (B, z_channels, H, W)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # 如果需要进一步将码本映射到嵌入空间，可以选择线性（Linear）或 MLP 投影器
        if args.use_cblinear == 1:
            print("****Using Linear Codebook Projector****")
            self.codebook_projection = torch.nn.Linear(codebook_dim, embed_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=embed_dim ** -0.5)
        elif args.use_cblinear == 2:
            print("****Using MLP Codebook Projector****")
            self.codebook_projection = torch.nn.Sequential(
                torch.nn.Linear(codebook_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, embed_dim),
            )
            # 注意：此处MLP的权重初始化可根据需要调整

        # 若量化器类型选择 "ema"，则需要额外的变量保存指数移动平均信息
        if self.quantize_type == "ema":
            self.decay = 0.99  # EMA 衰减因子
            self.eps = 1e-5    # 防止除零的数值稳定参数
            # cluster_size: 记录每个码本向量在当前 mini-batch 中被选中的次数，shape: (n_vision_words,)
            self.cluster_size = torch.nn.Parameter(torch.zeros(args.n_vision_words), requires_grad=False)
            # embed_avg: 每个码本向量的累积和，用于 EMA 更新，形状: (n_vision_words, embedding_dim)
            self.embed_avg = torch.nn.Parameter(self.tok_embeddings.weight.clone(), requires_grad=False)
            self.update = True  # 标记 EMA 更新是否进行
            self.tok_embeddings.weight.requires_grad = False  # EMA 方式下不通过反向传播更新码本
            self.num_tokens = args.n_vision_words  # 保存码本中总的词数

    # ----------------------------------------------------------------------------
    # 下方的一系列函数实现了模型的不同组成部分：
    # 1. hinge_d_loss：用于计算鉴别器的 hinge 损失
    # 2. calculate_adaptive_weight：通过比较梯度范数计算自适应损失权重
    # 3. cluster_size_ema_update / embed_avg_ema_update / weight_update：实现 EMA 更新码本的统计值
    # 4. quantize：核心量化函数，将连续潜在向量映射到离散码本，并计算量化损失
    # 5. forward：前向传播，依据训练阶段（生成器或鉴别器更新）计算不同的损失
    # 6. encode / decode：分别实现编码器和解码器的操作
    # 7. get_last_layer / decode_code：辅助函数，用于获取解码器末层和根据码本生成图像
    # ----------------------------------------------------------------------------

    def hinge_d_loss(self, logits_real, logits_fake):
        # 宏观目的：计算鉴别器损失，鼓励真实图像的判别值大于1，生成图像小于-1
        # logits_real: Tensor，形状通常为 (B, *)，代表真实图像的输出
        # logits_fake: Tensor，生成图像的输出
        loss_real = torch.mean(F.relu(1. - logits_real))  # 对真实图像的 hinge 损失，期望 logits_real >= 1
        loss_fake = torch.mean(F.relu(1. + logits_fake))  # 对生成图像的 hinge 损失，期望 logits_fake <= -1
        d_loss = 0.5 * (loss_real + loss_fake)  # 平均两个损失作为最终鉴别器损失
        return d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, discriminator_weight, last_layer=None):
        # 宏观目的：平衡重构损失 (nll_loss) 与生成器损失 (g_loss) 的权重
        # 通过比较 last_layer 上的梯度范数来自适应调整鉴别器损失的贡献，确保各项损失处于同一数量级
        # 其中:
        #   nll_loss: 重构相关损失（例如 L1 或 L2 损失）
        #   g_loss: 生成器的对抗损失（由鉴别器反馈）
        #   discriminator_weight: 用户设定的权重因子
        #   last_layer: 通常为解码器最后一层（例如 conv_out）的权重 tensor，形状与实际层权重一致
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]  # 求 nll_loss 对 last_layer 的梯度
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]  # 求 g_loss 的梯度

        # 计算梯度的 L2 范数（即各自张量元素的平方和开根号）
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()  # 限制 d_weight 范围，同时 detach 避免反向传播
        d_weight = d_weight * discriminator_weight  # 乘上鉴别器的权重因子
        return d_weight

    def cluster_size_ema_update(self, new_cluster_size):
        # 宏观目的：用 EMA 更新码本索引的累积统计，帮助稳定码本更新
        # new_cluster_size: 新的簇大小统计，Tensor 形状：(n_vision_words,)
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        # 宏观目的：更新码本向量对应的累积和，后续将归一化更新权重
        # new_embed_avg: 新的嵌入向量求和，Tensor 形状：(n_vision_words, embedding_dim)
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        # 宏观目的：根据 EMA 更新后的统计结果归一化码本得嵌入向量，然后复制到 tok_embeddings 中
        n = self.cluster_size.sum()  # 汇总所有码本向量累计使用次数，标量
        # smoothed_cluster_size: 将 cluster_size 归一化并平滑，形状：(n_vision_words,)
        smoothed_cluster_size = ((self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n)
        # 对 embed_avg 进行归一化，每个码本向量除以对应的 smoothed_cluster_size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)  # unsqueeze 将 (n,) 转为 (n,1)
        self.tok_embeddings.weight.data.copy_(embed_normalized)  # 更新嵌入权重

    def quantize(self, z, temp=None, rescale_logits=False, return_logits=False):
        """
        宏观目的：
        将编码器输出的连续张量 z 量化为离散码本向量，并计算量化损失。
        
        参数：
          z: Tensor，编码器输出经过 quant_conv 后的特征，形状为 (B, C, D, H, W)，其中 C==embed_dim==e_dim
          
        主要步骤：
          1. z经过 rearrange 后 shape 变为 [b, d, h, w, e_dim] （转换到最后一维为embedding维度）
          2. 将 z 展平为二维张量 (B*H*W*D, C)
          3. 计算每个 z 向量与所有码本向量之间的欧几里得距离平方
          4. 根据不同量化策略（例如 ema 或非 ema）进行对应处理
          5. 使用直通估计器（gradient trick）保留反向传播梯度
          6. 将量化后的张量恢复到原输入形状，并（若需要）对索引进行 remap
        """
        z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim) #(2400, z_channels=4)

        # 如果使用代码本投影，则将 tok_embeddings.weight 经过投影层映射到 embed_dim
        if self.args.use_cblinear != 0:
            tok_embeddings_weight = self.codebook_projection(self.tok_embeddings.weight)
        else:
            tok_embeddings_weight = self.tok_embeddings.weight

        # 计算每个 z 向量与所有码本向量之间的欧氏距离的平方
        # 公式：||z - e||^2 = ||z||^2 + ||e||^2 - 2 * (z · e)
        # z_flattened: (B*H*W, C)； tok_embeddings_weight: (n_tokens, C)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(tok_embeddings_weight ** 2, dim=1) - \
            2 * torch.einsum('bd,dn->bn', z_flattened, rearrange(tok_embeddings_weight, 'n d -> d n'))
        # 得到距离矩阵 d，形状: (B*H*W, n_tokens)

        # 对每个 z 向量选取距离最小的码本索引
        min_encoding_indices = torch.argmin(d, dim=1) 

        # 针对不同的量化策略：EMA 或 非 EMA
        if self.quantize_type == "ema": #! 未测试
            # 获取对应的量化结果 z_q: 通过 embedding 取出向量，重塑为 (B, D, H, W, C)
            z_q = self.tok_embeddings(min_encoding_indices).view(z.shape)
            # 构造 one-hot 编码矩阵，形状: (B*H*W, n_tokens)
            encodings = F.one_hot(min_encoding_indices, self.num_tokens).type(z.dtype)
            # 求各个码本被选中的平均概率，shape: (n_tokens,)
            avg_probs = torch.mean(encodings, dim=0)
            # 计算困惑度（perplexity），衡量码本的使用多样性
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))
            min_encodings = None
            # EMA 更新：累计每个码本向量的选取次数（encodings_sum shape：(n_tokens,)）
            encodings_sum = encodings.sum(0)
            self.cluster_size_ema_update(encodings_sum)
            # 计算当前 mini-batch 中，各码本向量对应的嵌入向量总和
            embed_sum = encodings.transpose(0,1) @ z_flattened  # shape: (n_tokens, embedding_dim)
            self.embed_avg_ema_update(embed_sum)
            # 根据 EMA 更新后的统计归一化 tok_embeddings 权重
            self.weight_update(self.num_tokens)
            # 计算量化损失，使用 MSE 损失，即 z_q 与 z 之间的均方误差（detach 保证非梯度传递）
            loss = F.mse_loss(z_q.detach(), z)
        else:
            # 非 EMA 情况下，直接根据最小编码索引取得嵌入向量
            min_encodings = None
            perplexity = None
            # 利用 F.embedding 获取结果，重塑回 (B, D, H, W, C)
            z_q = F.embedding(min_encoding_indices, tok_embeddings_weight).view(z.shape)
            # 量化损失：结合两部分损失，部分项使用 detach 保证梯度独立传播
            loss = torch.mean((z_q.detach() - z)**2) + 0.33 * torch.mean((z_q - z.detach()) ** 2)

        # 直通估计器技巧：使 z_q 在梯度反向传播时和 z 保持一致（实际上梯度来自 z）
        z_q = z + (z_q - z).detach()

        # 恢复张量形状为 (B, C, D, H, W)，与输入一致
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        # 如果定义了 remap，则对编码索引进行重塑及映射
        if self.remap is not None:
            # 先将 min_encoding_indices reshape 为 (B, -1)
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)
        
        # # 如果 sane_index_shape 为 True，则将索引 reshape 为 (B, H, W)
        # if self.sane_index_shape:
        #     min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])
        # 如果 sane_index_shape 为 True，则将索引 reshape 为 (B, D, H, W)
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        # 返回：量化后的张量 z_q，量化损失 loss，以及额外信息 (距离矩阵 d, min_encodings, 最终编码索引)
        return z_q, loss, (d, min_encodings, min_encoding_indices)
    
    def forward(self, input, args, global_input=None, data_iter_step=0, step=0, is_val=False):
        """
        宏观目的：
        前向传播函数，根据不同训练阶段和步骤计算以下内容：
          - 进行完整的 VQGAN 计算：编码->量化->解码，计算重构损失、感知损失、以及对抗损失
        参数：
          input: 输入图像，Tensor，形状 (B, C, H, W, D) = (B, 4, 60, 100, 20)
          global_input: 全局条件输入（比如用于条件生成，具体用途依任务而定）
          data_iter_step: 当前迭代步数，用于调度鉴别器更新时机
          step: 训练步骤标志（0 为生成器更新，非 0 时为鉴别器更新）
          is_val: 是否为验证模式，控制是否计算对抗损失
        """
        output_dict = {}
        # 先对输入图像进行编码和量化，返回：
        #  quant: 量化后的潜在表示，形状 (B, embed_dim, H', W')
        #  qloss: 量化损失（标量）
        #  tk_labels: 量化后对应的码本索引，原始形状为 (B*H'*W',) 或 (B, H', W') 依据 sane_index_shape
        quant, qloss, [_, _, tk_labels] = self.encode(input, args) 
        
        # 对量化后的特征进行解码，生成重构图像 dec，shape 应为 (B, 3, H, W)
        
        # dec = self.decode(quant) #! 原版解码器尚未实现

        # # 计算重构损失：均值 L1 损失（输入与重构图像的绝对误差的均值）
        # rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous())) 
        # # 计算感知损失：LPIPS，用于衡量两幅图像在深度特征空间的差异
        # # p_loss = torch.mean(self.perceptual_loss(input.contiguous(), dec.contiguous()))
        # p_loss = 0

        output_dict.update({'embed_loss': qloss})
        mid = quant
        logits = self.forward_decoder(quant, input.shape)
        # 输出字典中添加解码后生成的 logits 与中间表示
        output_dict.update({'logits': logits})
        output_dict.update({'mid': mid})
        # import pdb; pdb.set_trace()
        return output_dict
        # if step == 0:  # 生成器更新阶段
        #     # 通过鉴别器对生成的图像 dec 进行预测，获取 logits，通常形状为 (B, 1) 或 (B, N)
        #     logits_fake = self.discriminator(dec)
        #     g_loss = -torch.mean(logits_fake)  # 生成器希望提高鉴别器预测值，所以取负值

        #     if is_val:
        #         # 验证时不进行对抗损失更新，只返回各项损失的加权和
        #         loss = rec_loss + self.args.rate_q * qloss
        #         return loss, rec_loss, qloss, p_loss, g_loss, tk_labels.view(input.shape[0], -1), dec
            
        #     # 计算自适应权重，根据解码器最后一层的梯度比较生成器损失与重构损失的比例
        #     d_weight = self.calculate_adaptive_weight(
        #         rec_loss + self.perceptual_weight * p_loss,
        #         g_loss,
        #         self.args.rate_d,
        #         last_layer=self.decoder.conv_out.weight
        #     )
            
        #     # 根据迭代步数决定是否启用对抗损失项（例如 disc_start 之后）
        #     if data_iter_step > self.args.disc_start:
        #         loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + d_weight * g_loss
        #     else:
        #         loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + 0 * g_loss

        #     return loss, rec_loss, qloss, p_loss, g_loss, tk_labels, dec
        # else:  # 鉴别器更新阶段
        #     # 对输入真实图像和生成图像均使用 detach() 防止梯度回传到生成器
        #     logits_real = self.discriminator(input.contiguous().detach().clone())
        #     logits_fake = self.discriminator(dec.detach().clone())
        #     # 计算 hinge 损失，鼓励真实样本输出较高值，生成样本输出较低值
        #     d_loss = self.hinge_d_loss(logits_real, logits_fake)
        #     loss = d_loss + 0 * (rec_loss + qloss + p_loss)  # 鉴别器更新中只关心 d_loss

        #     return loss, rec_loss, qloss, p_loss, d_loss, tk_labels, dec


        

    # --------------------------------------------------
    # forward_encoder 方法：完成编码过程，将输入图像 x 映射到中间隐变量空间
    def forward_encoder(self, x):
        # 输入 x 的形状为 (batch_size=1, channels=4, height=60, width=100, depth=20)
        bs, C, H, W, D = x.shape
        # 将 x 重排：将通道数与深度维度合并，变为 (bs, H, W, C*D=80)
        x = rearrange(x, 'b c h w d -> b h w (c d)')
        # 使用 embedder 进行线性投影，将每个像素的 (C*D=80) 维向量映射到 mid_channels=320 维度
        x = self.embedder(x) # x 形状为 (bs, H, W, mid_channels=320)
        # 将维度/permutation调整为 (batch, mid_channels, H, W) 后添加一个深度维度
        x = x.permute(0, 3, 1, 2).unsqueeze(-1) # x 的形状为: (batch_size=1, mid_channels=320, height=60, width=100, depth=1)
        # 传入 encoder_gpt 后，再经过 pre_vq_conv 将通道数映射到 z_channels
        # import pdb; pdb.set_trace()
        x = self.encoder_gpt(x) #(1,320,30,50,1)
        x = self.pre_vq_conv(x) #(1,4,30,50,1)
        # x = self.pre_vq_conv(self.encoder_gpt(x)) # 使用 1×1×1 的卷积（相当于逐通道的线性变换）将通道数从 mid_channels 转换为目标通道数 z_channels，且保证卷积前后空间尺寸不变。
        # 返回的 x 即为编码后的中间表示（用于后续量化）
        return x
    # 目的：将原始输入图像经过线性投影和卷积编码，得到量化前的中间潜变量表示

    def forward_decoder(self, z, input_shape):
        # 先将 z 通过 post_vq_conv 将通道数映射回 mid_channels
        # 再通过 decoder_gpt 进行解码处理，复原空间尺寸和特征表达
        x = self.decoder_gpt(self.post_vq_conv(z))
        # 根据传入的 input_shape 获取原始输入的尺寸 (bs, C, H, W, D)
        bs, C, H, W, D = input_shape
        # 将 x 的维度进行调整：先 permute 成 (bs, H, W, D, mid_channels)
        # 再利用 embedder_t 将 mid_channels 转换回输出通道数 out_channels，
        # 这里的 embedder_t 可以理解为映射每个像素的特征到 RGB 或其他目标特征
        similarity = self.embedder_t(x.permute(0, 2, 3, 4, 1))
        # 对 similarity 进行 rearrange 操作，以确保输出形状符合 (bs, out_channels, H, W, D)
        # 注意这里使用了 'b h w dd (c d) -> b c h w (dd d)' 的变换，其中 dd 固定为 1，d 对应原始D
        similarity = rearrange(similarity, 'b h w dd (c d) -> b c h w (dd d)', dd=1, d=D)
        # 返回解码后最终映射到输出通道的图像
        return similarity

    def encode(self, input, args):
        """
        宏观目的：
        1. 使用编码器将图像输入转换为中间特征表示
        2. 用量化卷积层 (quant_conv) 将特征转换到码本嵌入维度上
        3. 调用量化函数将连续特征映射到离散码本上，输出量化后张量和量化损失
          
        参数：
          input: 输入图像 Tensor，形状 (B, C=4, H=60, W=100, D=20)
        返回：
          quant: 量化后的潜在向量，形状 (B, embed_dim, H', W')
          emb_loss: 量化损失，标量 Tensor
          info: 其它信息（例如距离矩阵、编码索引等）
        """
        if args.encoder_type == 'vqgan':
            # 1. 编码：将输入经过 encoder 获取中间表示 h
            h = self.forward_encoder(input) # h.shape: torch.Size([1, 4, 30, 50, 1])

        elif args.encoder_type == 'vqgan_lc': #! 原版编码器尚未实现 #todo
            h = self.encoder(input)  # 编码器输出，形状依 ddconfig 决定，通常为 (B, z_channels, H', W')
            h = self.quant_conv(h)  # 通过量化卷积映射到嵌入空间，输出形状 (B, embed_dim, H', W')
            if self.e_dim == 768 and self.args.tuning_codebook != -1:
                # 若 embed_dim 为768，则对特征进行 L2 归一化，归一化后每个特征向量的 L2 范数为 1
                h = h / h.norm(dim=1, keepdim=True) 
        
        quant, emb_loss, info = self.quantize(h)  # 调用量化函数 quant.shape=(B,4,30,50,1)
        return quant, emb_loss, info

    def decode(self, quant, global_c_features=None):
        """
        宏观目的：
        将量化后的潜在向量映射回原始图像空间
        步骤：
          1. 先通过 post_quant_conv 将嵌入向量映射到编码器对应的通道数
          2. 再通过解码器还原成图像
          
        参数：
          quant: 量化后的张量，形状 (B, embed_dim, H', W')
        返回：
          dec: 解码还原后的图像，形状 (B, 3, H, W)
        """
        quant = self.post_quant_conv(quant)  # 映射回 z_channels，形状: (B, z_channels, H', W')
        dec = self.decoder(quant)  # 解码器还原图像
        return dec
    
    def get_last_layer(self):
        # 返回解码器最后一层卷积层的权重，用于自适应权重计算
        return self.decoder.conv_out.weight

    def decode_code(self, code_b):
        # 根据给定的码本索引张量 code_b，利用量化层中的 embedding 转换为嵌入向量，再解码为图像
        quant_b = self.quantize.embedding(code_b)
        dec = self.decode(quant_b)
        return dec
    


if __name__ == "__main__":
    # 定义一个简单的参数类
    class DummyArgs:
        pass

    args = DummyArgs()
    args.encoder_type = "vqgan"
    args.quantizer_type = "default"   # 非 EMA 情况
    args.tuning_codebook = -1         # 随机初始化且可调
    args.n_vision_words = 1000
    args.local_embedding_path = ""
    args.use_cblinear = 2
    args.rate_p = 0.0
    args.disc_start = 0
    args.rate_q = 1.0
    args.rate_d = 1.0

    # 设定模型参数：
    # - 输入通道：80
    # - 最终输出通道：80
    # - 中间通道：320
    # - 潜变量通道（z_channels）：4
    # - 嵌入向量维度 embed_dim：4（与 z_channels 相同）
    model = VQModel(args, inp_channels=80, out_channels=80, mid_channels=320, z_channels=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 测试输入：随机生成一个张量，形状 (B, C, H, W, D) = (batch_size, 4, 60, 80, 20)
    batch_size = 2
    test_input = torch.rand(batch_size, 4, 60, 100, 20).to(device)

    with torch.no_grad():
        output = model(test_input)

    # 打印输出各部分的形状
    # 输出字典包含：'embed_loss' (量化损失标量)，'logits' (解码后的输出)，'mid' (encoded tensor)
    print("Output keys:", list(output.keys()))
    print("embed_loss:", output["embed_loss"])
    print("logits shape:", output["logits"].shape)  # 预期形状: (B, out_channels, H, W, D) 即 (2, 4, 60, 80, 20)
    print("mid shape:", output["mid"].shape)        # 预期形状与编码器输出一致

    print("Done!")