# ===== 标准库 =====
import os
import math
import logging
import argparse
import pickle
from pathlib import Path

# ===== 第三方库 =====
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===== PyTorch =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

# ===== 自定义模块 =====
from vqvae.vae_2d_resnet import VAERes2DImg, VAERes2DImgDirectBC, VectorQuantizer, VAERes3DImgDirectBC
from vqganlc.models.vqgan_lc import VQModel
from training.pretrain_dataset import MyDatasetOnlyforVoxel, MyDatasetOnlyforVAE
from training.train_arg_parser import get_args_parser
from model import LSSTPVDAv2OnlyForVoxel, LSSTPVDAv2
from vismask.gen_vis_mask import sdf2occ
from training import distributed_mode
from training.schedulers import WarmupCosineLRScheduler, SchedulerFactory
from diffusers import AutoencoderKLCogVideoX

logger = logging.getLogger(__name__)


def my_collate_fn(batch):
    """
    batch 是一个列表，每个元素形如 (imgs_tensor, [img_meta])
    我们对 imgs_tensor 使用 torch.stack 进行堆叠，
    而对元数据则直接提取（注意这里假设每个样本返回的是一个只有一个 dict 的 list）。
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    # 由于 __getitem__ 返回的是 [img_meta]，因此取出列表内的第一个元素
    return imgs


def save_checkpoint(save_path, epoch, vqvae, optimizer, lr_scheduler):
    # save_checkpoint: 用于将当前模型和优化器的状态保存到本地文件，
    # 以实现断点续训或事后analysis。
    # 准备保存字典，包含模型参数（vqvae.state_dict())、优化器参数、lr调度器参数、当前epoch等信息
    checkpoint = {
        'epoch': epoch,
        'vqvae_state_dict': vqvae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
    }
    torch.save(checkpoint, save_path)
    print(f"=> Saved checkpoint to {save_path}")

# load_checkpoint: 在需要断点续训时，从已有的 checkpoint 文件中恢复训练的进度、模型参数、优化器参数等。
def load_checkpoint(ckpt_path, vqvae, optimizer, lr_scheduler):
    # 从 ckpt_path 加载 checkpoint 字典，然后恢复 epoch、模型参数、优化器参数等
    if not os.path.isfile(ckpt_path):
        print(f"=> No checkpoint found at '{ckpt_path}'")
        return 1  # 若没有找到checkpoint文件则从第1epoch开始
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1  # 下次从这个epoch继续训练
    vqvae.load_state_dict(checkpoint['vqvae_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler and checkpoint['lr_scheduler_state_dict'] is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    print(f"=> Loaded checkpoint from '{ckpt_path}', start_epoch {start_epoch}")
    return start_epoch


def train_vae(args, vqvae, train_loader, val_loader, device):
    vqvae = vqvae.module if isinstance(vqvae, torch.nn.parallel.DistributedDataParallel) else vqvae
    optimizer = torch.optim.AdamW(vqvae.parameters(), lr=args.lr, betas=(0.9, 0.999),weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    # warmup_steps = int(0.05 * total_steps)  # 5% warmup
    warmup_steps = 500  # 500 warmup

    if args.use_scheduler:
        factory = SchedulerFactory(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=args.lr,
            min_lr=1e-6
        )
        lr_scheduler = factory.get(name="cosine")  # 返回 SchedulerWrapper
    else:
        lr_scheduler = None

    # 尝试加载 checkpoint（若 args.resume 为 True，且在 args.resume_ckpt 中指定了 ckpt 文件路径）
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume_ckpt, vqvae, optimizer, lr_scheduler)
    
    vqvae.train()

    # 主训练循环，从 start_epoch 到 args.epochs，每个epoch遍历一遍 train_loader
    for epoch in range(start_epoch, args.epochs + 1):
        # 若使用了分布式Sampler，需要设置当前epoch，以便它做好shuffle
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for step, imgs in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            

            # =====================
            # 1) 3DVAE前向传播
            # (1) encode => 得到后验分布
            # (2) reparameterize (sampling)
            # (3) decode => 重构输出
            # todo 选择先在此处对 imgs 做零填充(F.pad) 以保证 /8 整除
            encode_out = vqvae.encode(imgs, return_dict=True) #todo
            posterior = encode_out.latent_dist
            kl_loss = posterior.kl().mean()  # 标准KL散度损失(相对于N(0,1))

            # reparameterize: 这里采用随机采样
            z = posterior.sample()
            # print(z.shape) # ([1, 4, 1, 65, 98])

            # 解码
            decode_out = vqvae.decode(z, return_dict=True)
            recons = decode_out.sample  # [B, 4, 3, h', w'] (按实际网络结构可能会包含pad)
            # print(recons.shape) # ([1, 4, 3, 520, 784])

            # =====================
            # 2) 计算loss并反向传播
            # =====================
            #todo 该任务对 Depth 渠道或者多帧时序还有专门的损失方式（例如对 Depth 使用别的指标），也可在此处把 recon_loss 拆分成 RGB 重构损失、Depth 重构损失、以及时域上的一致性损失等等
            # import pdb; pdb.set_trace()
            recon_loss = F.mse_loss(recons, imgs) # todo 这里有错误吧，recons的shape是（B，4，1，H，W）而imgs的shape是（B，4，3，H，W），应该都是（B，4，3，H，W）
            # todo VAE 训练中，KL 损失容易出现“塌陷”或“KL 消失”的问题
            total_loss = recon_loss + kl_loss  # 最简单的 VAE 损失: recon_loss + KL

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_lr()
            else:
                current_lr = args.lr
            
            # 打印或记录训练日志
            if step % args.log_interval == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} "
                      f"ReconLoss: {recon_loss.item():.6f} "
                      f"KL: {kl_loss.item():.6f} "
                      f"TotalLoss: {total_loss.item():.6f} "
                      f"LR: {current_lr:.6f}")

            # 中途进行验证
            # if step % args.val_interval == 0 and val_loader is not None:
            #     validate_vae(args, vqvae, val_loader, device, epoch, step)

            # 定期保存checkpoint
            if step % args.save_interval == 0:
                save_path = os.path.join(args.checkpoint_dir, f"vqvae_epoch{epoch}_step{step}.pth")
                save_checkpoint(save_path, epoch, vqvae, optimizer, lr_scheduler)
        
        if args.save_end_of_epoch:
            save_path = os.path.join(args.checkpoint_dir, f"vqvae_epoch{epoch}.pth")
            save_checkpoint(save_path, epoch, vqvae, optimizer, lr_scheduler)           
    
    print("Training Finished!")


# ==============================
def main():

    parser = argparse.ArgumentParser("Train VQ-VAE/VAE Model")
    parser.add_argument("--dist_on_itp",
                        action="store_true",
                        default=False,
                        help="使用基于 ITp 的分布式模式（如 MPI 的 OMPI 环境） if set.")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="分布式训练中当前进程的全局排名（rank）.")
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="参与分布式训练的总进程数.")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="当前进程使用的 GPU id.")
    parser.add_argument("--dist_backend",
                        type=str,
                        default="nccl",
                        help="分布式后端（推荐使用 'nccl'）.")
    parser.add_argument("--model", type=str, default="VQModel", help="choices: VAERes2DImgDirectBC, VQModel, Cog3DVAE")
    parser.add_argument("--mode", type=str, default="train", help="choices: train, eval")
    parser.add_argument("--dataset", type=str, default="MyDatasetOnlyforVoxel", help="Dataset name for logging")
    parser.add_argument("--validate_path", type=str, default='./output/d408_1024', help="12")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=8, help="number of data loading workers")
    parser.add_argument("--pin_mem", action='store_true', help="use pin memory in DataLoader")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_408", help="Where to save checkpoints")
    parser.add_argument("--resume", action='store_true', help="resume training from ckpt")
    parser.add_argument("--resume_ckpt", type=str, default="", help="which ckpt to resume from")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for VQ-VAE") # 原版4.5e-6
    parser.add_argument("--epochs", type=int, default=2, help="total epochs to train VQ-VAE")
    parser.add_argument("--save_interval", type=int, default=1000, help="save ckpt every n steps")
    parser.add_argument("--val_interval", type=int, default=2000, help="run validation every n steps")
    parser.add_argument("--log_interval", type=int, default=10, help="print log every n steps")
    parser.add_argument("--save_end_of_epoch", action='store_false', help="save checkpoint at each epoch end") #flag
    parser.add_argument("--use_scheduler", action='store_false', help="use StepLR scheduler or not")
    parser.add_argument("--max_val_steps", type=int, default=5, help="max batch for val step")
    # 以下是模型参数，与VQ-VAE定义对应（示例）
    parser.add_argument("--input_height", type=int, default=518,
                        help="图像输入高度")
    parser.add_argument("--input_width", type=int, default=784,
                        help="图像输入宽度")
    parser.add_argument("--inp_channels", type=int, default=80, help="输入通道数")
    parser.add_argument("--out_channels", type=int, default=80, help="输出通道数")
    parser.add_argument("--mid_channels", type=int, default=320, help="隐藏层通道数") # 1024
    parser.add_argument("--z_channels", type=int, default=4, help="潜变量通道数") # 256
    # vqganlc
    parser.add_argument('--concat', action='store_true', help='...')

    args = parser.parse_args()
    print(args)
    # -------------------------------
    # 初始化分布式环境 (多机多卡 or 单机多卡)
    # -------------------------------
    distributed_mode.init_distributed_mode(args)
    local_rank = args.gpu
    torch.cuda.set_device(local_rank)
    
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataset_train = MyDatasetOnlyforVAE(args) 
    
    # 获取总进程数
    num_tasks = distributed_mode.get_world_size() # 1
    # 获取当前进程编号
    global_rank = distributed_mode.get_rank() # 0
    print(f"--------num_tasks:{num_tasks}, global_rank:{global_rank}-----------")
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True
    )
    train_loader = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=my_collate_fn,
    )
    logger.info("Intializing DataLoader")

    # ----------------------------------------------------------------------------------
    # 直接从训练集中取前200条做val
    # ----------------------------------------------------------------------------------
    small_subset_size = 200
    if len(dataset_train) > small_subset_size:
        dataset_val = torch.utils.data.Subset(dataset_train, range(small_subset_size))  # 取前200条做验证
        val_sampler = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        val_loader = DataLoader(
            dataset_val,
            sampler=val_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=my_collate_fn,
        )
        logger.info("Initialized a small validation set from training data")

    
    vqvae_cfg = None  # 如果有额外vqvae的配置，可在此传
    if args.model == "Cog3DVAE":
        vqvae = AutoencoderKLCogVideoX(
            in_channels=args.inp_channels,
            out_channels=args.out_channels,
            sample_height=args.input_height,
            sample_width=args.input_width,
            latent_channels=4,
            temporal_compression_ratio=4.0,
        )
    else:
        raise ValueError("未识别的模型类型: " + args.model)
    
    vqvae.to(device)
    print("训练的模型类型为:", args.model)

    if distributed_mode.get_world_size() > 1:
        vqvae = DDP(vqvae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) # 根据warning修正

    vqvae_total_params = sum(p.numel() for p in vqvae.parameters())
    vqvae_trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
    # print(f"[Model Param Count] LSSTPVDAv2OnlyForVoxel total: {model_total_params}, trainable: {model_trainable_params}")
    print(f"[Model Param Count] VQ-VAE total: {vqvae_total_params}, trainable: {vqvae_trainable_params}")
    
    train_vae(args, vqvae, train_loader, val_loader, device)

    print("Done!")


if __name__ == "__main__":
    main()