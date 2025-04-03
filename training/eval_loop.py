# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import PIL.Image

import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN
import numpy as np
import time
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn

logger = logging.getLogger(__name__)
PRINT_FREQUENCY = 50


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: dict
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: nn.Module,                             # 传入评估对象模型，类型为 nn.Module
    model_input: nn.Module,                       # 另一个辅助模型/模块，用于生成条件信息或处理输入
    data_loader: Iterable,                        # 数据加载器，用于遍历评估数据集
    device: torch.device,                         # 指定设备，如 GPU 或 CPU
    epoch: int,                                   # 当前评估所在的 epoch（训练周期）
    fid_samples: int,                             # 用于FID评估的样本数量
    args: Namespace,                              # 其他参数配置对象
):
    gc.collect()  # 回收垃圾，清理内存，防止内存泄露或内存不足
    cfg_scaled_model = CFGScaledModel(model=model)  # 将传入模型进行 CFG（Classifier-Free Guidance）缩放包装
    cfg_scaled_model.train(False)  # 设置模型为评估模式（关闭 dropout、batchnorm更新等）

    # 根据 discrete_flow_matching 参数选择不同的采样器（solver）
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)   # 使用多项式调度器，参数 n=3.0 控制曲线形状
        path = MixtureDiscreteProbPath(scheduler=scheduler)  # 构建离散概率路径，利用调度器生成轨迹
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)  # 初始化一个概率分布向量（257维）
        p[256] = 1.0  # 将最后一个位置的概率置为1，表示特定的类别或状态（通常与 mask token 有关）
        solver = MixtureDiscreteEulerSolver(  # 使用混合离散欧拉求解器进行离散采样
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,       # 词汇表大小，与 p 的维数对应
            source_distribution_p=p,   # 初始的类别概率分布
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)  # 使用常规的常微分方程（ODE）求解器进行连续采样

    ode_opts = args.ode_options  # 读取ODE相关的参数选项

    num_synthetic = 0  # 已合成样本数量初始化为0
    snapshots_saved = False  # 快照是否保存的标志（当前未使用，但可能用于后续扩展）
    
    # 如果指定了输出目录，则创建快照文件夹用于保存中间评估结果
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    # 遍历数据加载器中的数据，每次获取一个 batch 数据
    for data_iter_step, samples in enumerate(data_loader):

        # 如果当前合成样本数量还未达到指定的 fid_samples 数量，则继续采样
        if num_synthetic < fid_samples:

            # 将 batch 中所有 tensor 数据移到指定设备上，加速计算
            for k, v in samples.items():
                samples[k] = samples[k].to(device, non_blocking=True)  # 非阻塞方式传输

            # 利用辅助模型，获取条件信息和辅助控制信息，确保不计算梯度
            with torch.no_grad():
                z, c, z_control = model_input(samples)  # 通过 model_input 生成潜变量 z、条件信息 c 和控制因子 z_control
                labels = {'c': c, 'z_control': z_control}  # 构造一个标签字典，后续采样时传入

            # 重置位移求解部分的函数调用次数计数器（nfe: number of function evaluations）
            cfg_scaled_model.reset_nfe_counter()

            if args.discrete_flow_matching:
                # ======= 离散采样部分 =======
                # 使用离散采样方式进行生成
                x_0 = (torch.zeros(z.shape, dtype=torch.long, device=device)  # 构造初始输入，所有位置全为0
                       + MASK_TOKEN)  # 并加上 MASK_TOKEN（代表掩码标记，通常为一个预定义的常量）
                
                # 根据配置决定如何构造对称函数，可能用于调整采样分布
                if args.sym_func:
                    sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)  # 自定义对称函数
                else:
                    sym = args.sym  # 直接从配置中获取函数

                # 根据传入的字符串设置采样数据类型
                if args.sampling_dtype == "float32":
                    dtype = torch.float32
                elif args.sampling_dtype == "float64":
                    dtype = torch.float64

                # 调用离散欧拉求解器进行采样生成合成样本
                synthetic_samples = solver.sample(
                    x_init=x_0,                                # 离散采样的初始数据
                    step_size=1.0 / args.discrete_fm_steps,     # 步长，根据采样步数设置
                    verbose=False,                              # 是否打印详细信息
                    div_free=sym,                               # 用于调整采样演化的对称函数
                    dtype_categorical=dtype,                    # 离散采样的数值类型
                    label=labels,                               # 条件标签，控制生成结果
                    cfg_scale=args.cfg_scale,                   # CFG scale参数，用于引导生成效果
                )
            else:
                # ======= 连续采样部分 =======
                # 对于连续采样，初始样本从标准正态分布中随机生成
                x_0 = torch.randn(z.shape, dtype=torch.float32, device=device)

                # 决定时间离散方法，若启用 EDM 调度则生成非均匀时间网格，否则为简单线性网格
                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_opts["nfe"])  # 根据 nfe 设置时间离散
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                # 这里使用 no_grad 确保在采样过程中不会计算梯度，节省内存和计算资源
                with torch.no_grad():
                    # 多次采样迭代，可能为了收敛或者采样多样性（此处循环10次）
                    for i in range(10):

                        # 调用求解器进行连续采样
                        synthetic_samples = solver.sample(
                            time_grid=time_grid,             # 时间网格，指明采样的时间段
                            x_init=x_0,                      # 初始输入
                            method=args.ode_method,          # 采样所使用的数值方法（如欧拉法、Runge-Kutta等）
                            return_intermediates=False,      # 是否返回中间步骤， False 表示仅最终结果
                            atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,  # 绝对误差容忍度
                            rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,  # 相对误差容忍度
                            step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,  # 步长设置
                            label=labels,                    # 条件标签
                            cfg_scale=args.cfg_scale,        # 引导尺度参数
                        )
                        labels['z_control'] = synthetic_samples.clone()  # 更新条件标签中的控制信息

                        # 对采样结果进行重排，按照视频帧、通道、空间维度重新组织tensor形状
                        synthetic_samples = rearrange(
                            synthetic_samples, 
                            'b f c h (w d) -> (b f) c h w d', 
                            f=model_input.num_frames, d=1
                        )
                        import pdb; pdb.set_trace()
                        # 使用输入辅助模型中的解码器进行解码生成“bbox”头部的输出
                        synthetic_samples = model_input.pts_bbox_head.vqvae.forward_decoder(  #!
                            synthetic_samples, (synthetic_samples.shape[0], 4, 60, 100, 20)
                        )
                        # 对 lidar 相关的数据也进行重排，确保形状匹配，此处 lidar2cams/lidar2imgs 可能代表传感器参数
                        lidar2cams = rearrange(samples['lidar2cams'], 'b (f n) h w -> (b f) n h w', f=model_input.num_frames)
                        lidar2imgs = rearrange(samples['lidar2imgs'], 'b (f n) h w -> (b f) n h w', f=model_input.num_frames)
                        # 利用辅助模块对采样结果解码为 sdf（signed distance function）输出结果
                        outs = model_input.pts_bbox_head.decode_sdf(
                            synthetic_samples, lidar2cams, lidar2imgs
                        )

                        # 循环保存输出结果的可视化文件，保存深度图和 occ map 可视化结果
                        for j in range(outs.shape[0]):
                            # ----- 保存 occ map -----
                            sdf = synthetic_samples[j, 0].cpu().numpy()  # 获取第 j 个样本的 sdf 数据，转换到 CPU
                            mask = np.load('/mnt/bn/occupancy3d/workspace/lzy/Occ3d/projects/mmdet3d_plugin/surroundocc/gen_vismask/mask_0.2.npy')
                            sdf = sdf2occ(sdf) * mask  # 将 sdf 数据转换为 occupancy 表示，再与 mask 相乘
                            root_dir = os.path.join(args.output_dir, "depth", str(data_iter_step))
                            os.makedirs(root_dir, exist_ok=True)  # 确保保存目录存在
                            save_dir = os.path.join(root_dir, f'{str(i * model_input.num_frames + j).zfill(3)}_sdf.png')
                            cv2.imwrite(save_dir, cv2.rotate(sdf.max(-1), cv2.ROTATE_180) * 255)  # 保存图片

                            # ----- 保存 depth 信息 -----
                            out = outs[j].view(3, 518 // 2, 784 // 2)  # 调整输出尺寸，将 tensor reshape 为 (3, H, W)
                            for k in range(3):  # 分别保存三个通道的深度图
                                save_dir = os.path.join(root_dir, f'{str(i * model_input.num_frames + j).zfill(3)}_{k}.png')
                                plt.imsave(save_dir, np.log10(out[k].cpu().numpy()), cmap='jet')  # 使用 logarithm 显示，采用 jet 彩色映射

            # 更新总共合成的样本数量
            num_synthetic += synthetic_samples.shape[0]
        
        else:
            # 一旦合成样本数量达到 fid_samples，结束采样并返回空字典（也可返回评估指标）
            return {}

        # 如果设置为测试运行模式，则结束评估提前退出
        if args.test_run:
            return {}


def sdf2occ(sdf):
    sdf[sdf == 0] = -1
    sdf[sdf > 0] = 0
    sdf[sdf < 0] = 1
    sdf[:1] = 0
    sdf[:, :1] = 0
    sdf[:, -1:] = 0
    sdf[:, :, :1] = 0
    sdf[:, :, -1:] = 0
    return sdf