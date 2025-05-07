#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
该脚本示例展示如何分别统计各个模型的参数量，包括总参数量和可训练参数量。
注意：
  1. 请确保脚本中的各个模型 (VAERes2DImgDirectBC, VAERes3DImgDirectBC, VQModel, AutoencoderKLCogVideoX)
     能够根据你的工程目录正确导入，如下所示。
  2. 如果你只想统计部分模型，可注释掉不需要的部分。
"""

import torch

# ===== 导入你的模型 =====
# 请根据你项目中的实际路径确认下面这些模块的导入是否正确
from vqvae.vae_2d_resnet import VAERes2DImgDirectBC, VAERes3DImgDirectBC
from vqganlc.models.vqgan_lc import VQModel
from diffusers import AutoencoderKLCogVideoX  # Cog3DVAE 示例模型

# 可以定义一个简单的函数快速统计参数量
def count_parameters(model):
    """
    统计模型的总参数量和可训练参数量

    Args:
        model (torch.nn.Module): 模型实例

    Returns:
        total_params (int): 模型所有参数数量
        trainable_params (int): 可训练参数的数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# 构造一个简单的配置对象，模拟 argparse 中的 args
class DummyArgs:
    pass

args = DummyArgs()
args.inp_channels = 80
args.out_channels = 80
args.mid_channels = 320
args.z_channels = 4
# 如果需要可传入额外的vqevae配置，否则为 None
vqvae_cfg = None

# ------------------------------
# 1. 统计 VAERes2DImgDirectBC 的参数量
# ------------------------------
model_2d = VAERes2DImgDirectBC(
    inp_channels=args.inp_channels,
    out_channels=args.out_channels,
    mid_channels=args.mid_channels,
    z_channels=args.z_channels,
    vqvae_cfg=vqvae_cfg
)
total_params_2d, trainable_params_2d = count_parameters(model_2d)
print("VAERes2DImgDirectBC 参数量：")
print("  总参数量：{}".format(total_params_2d))
print("  可训练参数量：{}".format(trainable_params_2d))
print("-----------------------------------------------------------")

# ------------------------------
# 2. 统计 VAERes3DImgDirectBC 的参数量
# ------------------------------
model_3d = VAERes3DImgDirectBC(
    inp_channels=args.inp_channels,
    out_channels=args.out_channels,
    mid_channels=args.mid_channels,
    z_channels=args.z_channels,
    vqvae_cfg=vqvae_cfg
)
total_params_3d, trainable_params_3d = count_parameters(model_3d)
print("VAERes3DImgDirectBC 参数量：")
print("  总参数量：{}".format(total_params_3d))
print("  可训练参数量：{}".format(trainable_params_3d))
print("-----------------------------------------------------------")

# ------------------------------
# 3. 统计 VQModel 的参数量
# ------------------------------
try:
    model_vq = VQModel(
        args=args,
        inp_channels=args.inp_channels,
        out_channels=args.out_channels,
        mid_channels=args.mid_channels,
        z_channels=args.z_channels,
        vqvae_cfg=vqvae_cfg
    )
    total_params_vq, trainable_params_vq = count_parameters(model_vq)
    print("VQModel 参数量：")
    print("  总参数量：{}".format(total_params_vq))
    print("  可训练参数量：{}".format(trainable_params_vq))
except Exception as e:
    print("初始化 VQModel 时出错：", e)
print("-----------------------------------------------------------")

# ------------------------------
# 4. 统计 Cog3DVAE (这里采用 AutoencoderKLCogVideoX) 的参数量
# ------------------------------
try:
    model_cog = AutoencoderKLCogVideoX(
        in_channels=args.inp_channels,
        out_channels=args.out_channels,
        sample_height=518,   # 根据你的实际输入尺寸设定
        sample_width=784,
        latent_channels=4,
        temporal_compression_ratio=4.0,
    )
    total_params_cog, trainable_params_cog = count_parameters(model_cog)
    print("Cog3DVAE (AutoencoderKLCogVideoX) 参数量：")
    print("  总参数量：{}".format(total_params_cog))
    print("  可训练参数量：{}".format(trainable_params_cog))
except Exception as e:
    print("初始化 Cog3DVAE 时出错：", e)
print("-----------------------------------------------------------")

# 如果你的设备支持CUDA，可以把模型放到GPU上，同时查看初始显存占用情况
if torch.cuda.is_available():
    device = torch.device("cuda")
    # 示例：将模型放置到 GPU 上
    model_2d.to(device)
    current_alloc = torch.cuda.memory_allocated(device) / 1024 / 1024
    current_max_alloc = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    print("[GPU 显存] 当前分配：{:.2f} MB，最大分配：{:.2f} MB".format(current_alloc, current_max_alloc))