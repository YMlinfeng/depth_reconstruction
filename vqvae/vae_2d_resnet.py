""" adopted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py """
# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import numpy as np
import math
from .utils import shift_dim , view_range
import torch.nn.functional as F
from copy import deepcopy
from .attention import *
from .quantizer import VectorQuantizer
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
            # print(p, total_pad[::-1])
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        # print(f"Conv3d weight shape: {(out_channels, in_channels, *kernel_size)}")
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)
        # print("samepad conv end")

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
            print(f"Encoder,i:{i}")
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            print(i)
            self.convs.append(conv)
            n_times_downsample -= 1
        # print("+++++++++")
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)
        # print("+++++++++")
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


# 假设已经导入用于3D卷积（SamePadConv3d）、Encoder、Decoder 以及 VectorQuantizer 模块
# 同时假设已导入类似 einops.rearrange 的函数用于张量重排（这里假设已从 einops 导入）

# 定义一个用于处理2D图像的变分自编码器（VAE），构建基于残差结构的图像编码和解码
class VAERes2DImg(nn.Module):
    def __init__(
        self,
        inp_channels=3,      # 输入图像的通道数，默认RGB图像（3通道）
        out_channels=3,      # 输出图像通道数，通常与输入图像通道数一致
        mid_channels=256,    # 中间隐层表示的通道数，用于投射到更高维特征空间
        z_channels=4,        # 潜变量 z 的通道数（通常较小，用于压缩表示）
        vqvae_cfg=None,      # VQ-VAE的配置（当前版本中未使用该变量，可扩展）
    ):
        super().__init__()
        
        # pre_vq_conv: 用于将 Encoder 的输出（mid_channels）映射到潜变量 z 空间（z_channels）
        # 使用 SamePadConv3d，核大小为1，即1x1x1卷积，保证不改变尺寸
        self.pre_vq_conv = SamePadConv3d(mid_channels, z_channels, 1)
        
        # encoder_gpt: 编码器模块，将中间表示进行编码
        # 参数：输入通道 mid_channels，残差层数24，下采样率为 (2,2,1)，注意 in_channels 也为 mid_channels
        self.encoder_gpt = Encoder(mid_channels, 24, (2, 2, 1), in_channels=mid_channels)
        
        # post_vq_conv: 用于将经过 VQ-VAE 量化后的 z，再映射回中间表示空间
        self.post_vq_conv = SamePadConv3d(z_channels, mid_channels, 1)
        
        # decoder_gpt: 解码器模块，将中间表示还原，参数与 encoder 对应
        self.decoder_gpt = Decoder(mid_channels, 24, (2, 2, 1), out_channel=mid_channels)
        
        # embedder: 将输入单个像素（或向量）的通道（inp_channels）映射到中间表示（mid_channels）
        # 这里处理可能将图像每个像素点的值投射到高维空间
        self.embedder = nn.Linear(inp_channels, mid_channels)
        
        # embedder_t: 将中间表示转换为最终输出图像通道（out_channels），类似于最后的线性投射
        self.embedder_t = nn.Linear(mid_channels, out_channels)
        
        # vqvae: 向量量化模块，用于将连续的潜变量表示离散化
        # 参数中 n_e 为嵌入字典中向量的数量（此处与 mid_channels 保持一致），e_dim 为编码维度（z_channels）
        # beta 作为量化损失的权重，use_voxel 表示使用体素数据（此配置可能针对3D场景）
        self.vqvae = VectorQuantizer(
            n_e=mid_channels,      # 这里 n_e 默认为 mid_channels 的值（例如 256）
            # n_e=512,      # 这里 n_e 默认为 mid_channels 的值（例如 256）
            e_dim=z_channels,    # e_dim 默认为 z_channels 的值（例如 4）#!是不是太小了
            beta=1., 
            z_channels=z_channels, 
            use_voxel=True
        )
    
    # --------------------------------------------------
    # sample_z 方法：对给定的z向量进行高斯采样变换，
    # 通常用于 VAE 中 reparameterization trick，
    # 将 z 向量切分为均值（mu）和对数方差部分，计算采样后的 z。
    def sample_z(self, z):
        # 这里假设 z 的第1维大小为 2 * latent_dim，前半部分为 mu，后半部分为 log(var)
        dim = z.shape[1] // 2
        mu = z[:, :dim]              # 均值向量
        sigma = torch.exp(z[:, dim:] / 2)  # 标准差，通过对数方差转换得到
        eps = torch.randn_like(mu)     # 随机噪声 epsilon
        # 通过 reparameterization trick 生成采样的 z
        return mu + sigma * eps, mu, sigma

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

    # --------------------------------------------------
    # forward_decoder 方法：将量化后的 z 解码还原回中间特征，然后转换形成最终输出图像
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
    # 目的：将量化之后的 z 经过解码器还原为中间特征，并由线性层映射为输出图像

    # --------------------------------------------------
    # forward 方法：整体前向传播流程
    # 1. 通过 forward_encoder 获得中间隐变量表示 z
    # 2. 使用 VQ-VAE 模块对 z 进行向量量化，获得离散化的 z_sampled 与量化损失
    # 3. 通过 forward_decoder 将量化后的 z 解码成 logits
    # todo：找到codebook的大小，并且想办法把codebook变大
    def forward(self, x, **kwargs): # x.shape: torch.Size([1, 4, 60, 100, 20])
        output_dict = {}  # 用于存放各个步骤的输出结果
        
        # 1. 编码：将输入经过 encoder 获取中间表示 z
        z = self.forward_encoder(x) # z.shape: torch.Size([1, 4, 30, 50, 1])
        
        # 2. VQ-VAE 模块，对连续的 z 值进行向量量化
        # 返回 quantized 的 z（即离散表示），量化损失 loss，以及其它信息 info（例如使用次数等）
        z_sampled, loss, info = self.vqvae(z, is_voxel=False) # z_sampled.shape: torch.Size([1, 4, 30, 50, 1])
        # 将量化过程中计算的损失添加到输出字典中，用于训练过程中的监督
        output_dict.update({'embed_loss': loss})
        
        mid = z_sampled  # 可选地保存中间量化表示
        # 3. 解码：将量化后的 z 解码得到 logits，代表重构的图像
        logits = self.forward_decoder(z_sampled, x.shape) # logits.shape: torch.Size([1, 4, 60, 100, 20])
        
        # 输出字典中添加解码后生成的 logits 与中间表示
        output_dict.update({'logits': logits})
        output_dict.update({'mid': mid})
        # import pdb; pdb.set_trace()
        return output_dict
    # 目的：整合编码、量化和解码流程，最终输出重构图像（或其相应的特征映射）以及量化损失

    # --------------------------------------------------
    # generate 方法：给定潜变量 z 和输入形状，执行生成过程
    # 与 forward_decoder 保持一致，用于推理阶段生成图像
    def generate(self, z, input_shape):
        logits = self.forward_decoder(z, input_shape)
        return {'logits': logits}
    # 目的：推理模式下直接从给定的 z 生成对应的图像输出



# 直接扩大codebook的大小
class VAERes2DImgDirectBC(nn.Module):
    def __init__(
        self,
        inp_channels=3,      # 输入图像的通道数，默认RGB图像（3通道）--80
        out_channels=3,      # 输出图像通道数，通常与输入图像通道数一致 --80
        mid_channels=256,    # Encoder的输出通道，用于投射到更高维特征空间 -- 1024
        z_channels=4,        # 潜变量 z 的通道数（通常较小，用于压缩表示）-- 256
        vqvae_cfg=None,      # VQ-VAE的配置（当前版本中未使用该变量，可扩展）
    ):
        super().__init__()
        
        # pre_vq_conv: 用于将 Encoder 的输出（mid_channels）映射到潜变量 z 空间（z_channels）
        # 使用 SamePadConv3d，核大小为1，即1x1x1卷积，保证不改变尺寸
        self.pre_vq_conv = SamePadConv3d(mid_channels, z_channels, 1)
        
        # 参数：输入通道 mid_channels，残差层数24，下采样率为 (2,2,1) #通道维度从inp到mid
        self.encoder_gpt = Encoder(mid_channels, 24, (2, 2, 1), in_channels=mid_channels) #!
        
        print('mid_channels:', mid_channels)
        # post_vq_conv: 用于将经过 VQ-VAE 量化后的 z，再映射回中间表示空间
        self.post_vq_conv = SamePadConv3d(z_channels, mid_channels, 1)
        
        # decoder_gpt: 解码器模块，将中间表示还原，参数与 encoder 对应
        self.decoder_gpt = Decoder(mid_channels, 24, (2, 2, 1), out_channel=mid_channels)
        
        self.embedder = nn.Linear(inp_channels, mid_channels)
        
        # embedder_t: 将中间表示转换为最终输出图像通道（out_channels），类似于最后的线性投射
        self.embedder_t = nn.Linear(mid_channels, out_channels)
        
        # vqvae: 向量量化模块，用于将连续的潜变量表示离散化
        # 参数中 n_e 为嵌入字典中向量的数量（此处与 mid_channels 保持一致），e_dim 为编码维度（z_channels）
        # beta 作为量化损失的权重，use_voxel 表示使用体素数据（此配置可能针对3D场景）
        print(mid_channels)
        self.vqvae = VectorQuantizer( # * 离散化不仅可以压缩信息，还能够有效约束生成模型的潜在分布
            n_e=mid_channels,      # 这里 n_e 默认为 mid_channels 的值（例如 1024）
            # n_e=512,      # 这里 n_e 默认为 mid_channels 的值（例如 256）
            e_dim=z_channels,    # e_dim 默认为 z_channels 的值（例如 256）#! 应该必须是相等的
            beta=1., 
            z_channels=z_channels, # 编码器（Encoder）：首先将输入图像编码成一个低维的、隐空间（latent space）的表示。对于每个输入图像，编码器产生的输出特征图的通道数由配置参数 ddconfig["z_channels"] 定义，这里面的表示我们常称为“z”
            use_voxel=True
        )
    
    # --------------------------------------------------
    # sample_z 方法：对给定的z向量进行高斯采样变换，
    # 通常用于 VAE 中 reparameterization trick，
    # 将 z 向量切分为均值（mu）和对数方差部分，计算采样后的 z。
    def sample_z(self, z):
        # 这里假设 z 的第1维大小为 2 * latent_dim，前半部分为 mu，后半部分为 log(var)
        dim = z.shape[1] // 2
        mu = z[:, :dim]              # 均值向量
        sigma = torch.exp(z[:, dim:] / 2)  # 标准差，通过对数方差转换得到
        eps = torch.randn_like(mu)     # 随机噪声 epsilon
        # 通过 reparameterization trick 生成采样的 z
        return mu + sigma * eps, mu, sigma

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

    # --------------------------------------------------
    # forward_decoder 方法：将量化后的 z 解码还原回中间特征，然后转换形成最终输出图像
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
    # 目的：将量化之后的 z 经过解码器还原为中间特征，并由线性层映射为输出图像

    # --------------------------------------------------
    # forward 方法：整体前向传播流程
    # 1. 通过 forward_encoder 获得中间隐变量表示 z
    # 2. 使用 VQ-VAE 模块对 z 进行向量量化，获得离散化的 z_sampled 与量化损失
    # 3. 通过 forward_decoder 将量化后的 z 解码成 logits
    # todo：找到codebook的大小，并且想办法把codebook变大
    def forward(self, x, args, **kwargs): # x.shape: torch.Size([B, 4, 60, 100, 20])
        output_dict = {}  # 用于存放各个步骤的输出结果
        
        # 1. 编码：将输入经过 encoder 获取中间表示 z
        z = self.forward_encoder(x) # z.shape: torch.Size([1, 4, 30, 50, 1])
        
        # 2. VQ-VAE 模块，对连续的 z 值进行向量量化
        # 返回 quantized 的 z（即离散表示），量化损失 loss，以及其它信息 info（例如使用次数等）
        z_sampled, loss, info = self.vqvae(z, is_voxel=False) # z_sampled.shape: torch.Size([1, 4, 30, 50, 1])
        # 将量化过程中计算的损失添加到输出字典中，用于训练过程中的监督
        output_dict.update({'embed_loss': loss})
        
        mid = z_sampled  # 可选地保存中间量化表示
        # 3. 解码：将量化后的 z 解码得到 logits，代表重构的图像
        logits = self.forward_decoder(z_sampled, x.shape) # logits.shape: torch.Size([1, 4, 60, 100, 20])
        
        # 输出字典中添加解码后生成的 logits 与中间表示
        output_dict.update({'logits': logits})
        output_dict.update({'mid': mid})
        # import pdb; pdb.set_trace()
        return output_dict
    # 目的：整合编码、量化和解码流程，最终输出重构图像（或其相应的特征映射）以及量化损失

    # --------------------------------------------------
    # generate 方法：给定潜变量 z 和输入形状，执行生成过程
    # 与 forward_decoder 保持一致，用于推理阶段生成图像
    def generate(self, z, input_shape):
        logits = self.forward_decoder(z, input_shape)
        return {'logits': logits}
    # 目的：推理模式下直接从给定的 z 生成对应的图像输出


# 直接扩大codebook的大小
class VAERes3DImgDirectBC(nn.Module):
    def __init__(
        self,
        inp_channels=3,      # 输入图像的通道数，默认RGB图像（3通道）--80
        out_channels=3,      # 输出图像通道数，通常与输入图像通道数一致 --80
        mid_channels=256,    # Encoder的输出通道，用于投射到更高维特征空间 -- 1024
        z_channels=4,        # 潜变量 z 的通道数（通常较小，用于压缩表示）-- 256
        vqvae_cfg=None,      # VQ-VAE的配置（当前版本中未使用该变量，可扩展）
        height=5,
    ):
        super().__init__()
        
        self.height = height
        self.pre_vq_conv = SamePadConv3d(mid_channels * height, z_channels, 1)
        self.encoder_gpt = Encoder(mid_channels, 24, (2, 2, 4), in_channels=mid_channels) #!
        
        print('mid_channels:', mid_channels)
        self.post_vq_conv = SamePadConv3d(z_channels, mid_channels * height, 1)
        self.decoder_gpt = Decoder(mid_channels, 24, (2, 2, 4), out_channel=mid_channels)
        
        self.embedder = nn.Linear(inp_channels, mid_channels)

        self.embedder_t = nn.Linear(mid_channels, out_channels)
        print(mid_channels)
        self.vqvae = VectorQuantizer( # * 离散化不仅可以压缩信息，还能够有效约束生成模型的潜在分布
            n_e=mid_channels * height,      # 这里 n_e 默认为 mid_channels 的值（例如 1024）
            # n_e=512,      # 这里 n_e 默认为 mid_channels 的值（例如 256）
            e_dim=mid_channels * height,    # e_dim 默认为 z_channels 的值（例如 256）#! 应该必须是相等的
            beta=1., 
            z_channels=z_channels, # 编码器（Encoder）：首先将输入图像编码成一个低维的、隐空间（latent space）的表示。对于每个输入图像，编码器产生的输出特征图的通道数由配置参数 ddconfig["z_channels"] 定义，这里面的表示我们常称为“z”
            use_voxel=True
        )
    
    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]              # 均值向量
        sigma = torch.exp(z[:, dim:] / 2)  # 标准差，通过对数方差转换得到
        eps = torch.randn_like(mu)     # 随机噪声 epsilon
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        bs, C, H, W, D = x.shape
        x = rearrange(x, 'b c h w d -> b h w d c')
        x = self.embedder(x)
        x = rearrange(x, 'b h w d c -> b c h w d')
        x = self.encoder_gpt(x)
        x = rearrange(x, 'b c h w d -> b (c d) h w')
        x = self.pre_vq_conv(x.unsqueeze(-1))
        return x

    def forward_decoder(self, z, input_shape):
        z = self.post_vq_conv(z)
        z = rearrange(z, 'b (c d) h w dd -> b c h w (d dd)', d=self.height)
        x = self.decoder_gpt(z)
        similarity = self.embedder_t(x.permute(0, 2, 3, 4, 1))
        similarity = rearrange(similarity, 'b h w d c -> b c h w d')
        return similarity

    def forward(self, x, args, **kwargs): # x.shape: torch.Size([B, 4, 60, 100, 20])
        output_dict = {}  # 用于存放各个步骤的输出结果
        z = self.forward_encoder(x) # z.shape: torch.Size([1, 4, 30, 50, 1])
        
        z_sampled, loss, info = self.vqvae(z, is_voxel=False) # z_sampled.shape: torch.Size([1, 4, 30, 50, 1])
        output_dict.update({'embed_loss': loss})
        
        mid = z_sampled  # 可选地保存中间量化表示
        logits = self.forward_decoder(z_sampled, x.shape) # logits.shape: torch.Size([1, 4, 60, 100, 20])
        
        output_dict.update({'logits': logits})
        output_dict.update({'mid': mid})
        return output_dict

    def generate(self, z, input_shape):
        logits = self.forward_decoder(z, input_shape)
        return {'logits': logits}


if __name__ == "__main__":
    vae = VAERes3DVoxel().cuda()
    inp = torch.rand(2, 4, 60, 100, 20).cuda()
    outs = vae(inp)
    print('done')
