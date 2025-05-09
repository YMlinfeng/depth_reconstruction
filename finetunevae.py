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
import imageio

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
from training.pretrain_dataset import MyDatasetOnlyforVoxel, MyDatasetOnlyforVAE, MyDatasetforVAE3_3, MyDatasetforVAE_FRONT_1_9
from training.train_arg_parser import get_args_parser
from model import LSSTPVDAv2OnlyForVoxel, LSSTPVDAv2
from vismask.gen_vis_mask import sdf2occ
from training import distributed_mode
from training.schedulers import WarmupCosineLRScheduler, SchedulerFactory
from diffusers import AutoencoderKLCogVideoX

logger = logging.getLogger(__name__)


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


def load_checkpoint(ckpt_path, vqvae, optimizer, lr_scheduler):
    """
    加载模型训练的 checkpoint，包括模型参数、优化器状态和学习率调度器状态。

    参数:
        ckpt_path (str): checkpoint 文件路径
        vqvae (nn.Module): 待加载参数的模型实例
        optimizer (Optimizer): 优化器实例
        lr_scheduler (Scheduler): 学习率调度器实例（可为 None）

    返回:
        int: 加载后训练的起始 epoch
    """
    if not os.path.isfile(ckpt_path):
        print(f"=> ❌ No checkpoint found at '{ckpt_path}'")
        return 1  # 若没有找到 checkpoint 文件，则从第 1 epoch 开始

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    start_epoch = checkpoint.get('epoch', 0) + 1

    # 拿出权重字典
    state_dict = checkpoint.get('vqvae_state_dict', checkpoint)

    # 自动处理 'module.' 前缀匹配问题
    model_keys = list(vqvae.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if model_keys[0].startswith("module.") and not ckpt_keys[0].startswith("module."):
        # 模型是 DDP，但 checkpoint 是单卡保存的 → 添加 "module."
        new_state_dict = {"module." + k: v for k, v in state_dict.items()}
        print("✅ 加载权重：添加 'module.' 前缀")
    elif not model_keys[0].startswith("module.") and ckpt_keys[0].startswith("module."):
        # 模型是单卡，但 checkpoint 是 DDP 保存的 → 去除 "module."
        new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        print("✅ 加载权重：去除 'module.' 前缀")
    else:
        # 模型和 checkpoint 保持一致 → 无需改名
        new_state_dict = state_dict
        print("✅ 加载权重：无需修改前缀")

    # 加载参数
    vqvae.load_state_dict(new_state_dict, strict=True)
    print(f"=> 🧠 Loaded model weights from '{ckpt_path}'")

    # 加载优化器和调度器状态
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded.")
    else:
        print("⚠️ Warning: Optimizer state not found in checkpoint.")

    if lr_scheduler and checkpoint.get('lr_scheduler_state_dict', None) is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("✅ LR scheduler state loaded.")
    else:
        print("⚠️ Warning: LR scheduler state not found or not used.")

    return start_epoch

def visualize_sample(orig_tensor, recon_tensor, output_prefix, fps):
    """
    对单个样本进行可视化：
      - orig_tensor, recon_tensor 的shape均为 (C, T, H, W)
      - 输出视频：对每一帧将原始与重构帧横向拼接后生成视频文件
      - 输出图片：保存每一帧为单张图片，同时拼接成3x3的拼图保存
    参数：
      orig_tensor: 原始视频样本 tensor，范围为[-1,1]
      recon_tensor: 重构视频样本 tensor，范围为[-1,1]
      output_prefix: 保存时的文件名前缀（包括路径）
      fps: 视频帧率
    """

    # 确保 output_prefix 的父目录存在
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 断开梯度，转换 tensor 为 numpy 数组并调整顺序：从 (C, T, H, W) 到 (T, H, W, C)
    orig_np = orig_tensor.detach().cpu().permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    recon_np = recon_tensor.detach().cpu().permute(1, 2, 3, 0).numpy()

    # 将 [-1,1] 映射到 [0,255]
    orig_np = np.clip((orig_np + 1) / 2 * 255, 0, 255).astype(np.uint8)
    recon_np = np.clip((recon_np + 1) / 2 * 255, 0, 255).astype(np.uint8)

    # 对每一帧进行横向拼接：形成对比帧
    composite_frames = [np.concatenate([orig_np[i], recon_np[i]], axis=1) for i in range(orig_np.shape[0])]

    # ----------------------------------
    # 1) 保存视频：确保保存视频的目录存在，再写入视频文件
    # ----------------------------------
    video_path = output_prefix + "_comparison.mp4"
    video_dir = os.path.dirname(video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264')
    for frame in composite_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved comparison video to {video_path}")

    # ----------------------------------
    # 2) 保存各帧图片，并生成3x3拼图
    # ----------------------------------
    frames_folder = output_prefix + "_frames"
    os.makedirs(frames_folder, exist_ok=True)
    for i, frame in enumerate(composite_frames):
        frame_path = os.path.join(frames_folder, f"frame_{i:03d}.png")
        imageio.imwrite(frame_path, frame)
    print(f"Saved individual frame images to folder {frames_folder}")

    # 若帧数正好为9，则拼接成3x3的 collage 图
    if len(composite_frames) == 9:
        rows = []
        for i in range(3):
            row = np.concatenate(composite_frames[i*3:(i*3+3)], axis=1)
            rows.append(row)
        collage = np.concatenate(rows, axis=0)
        collage_path = output_prefix + "_collage.png"
        collage_dir = os.path.dirname(collage_path)
        if collage_dir:
            os.makedirs(collage_dir, exist_ok=True)
        imageio.imwrite(collage_path, collage)
        print(f"Saved collage image to {collage_path}")
        
# ---------------------------------------------------------
# 修改后的 validate_vae 函数：验证时保存视频和 9 张图片（拼图和单帧均保存）
# ---------------------------------------------------------
def validate_vae(args, vae, val_loader, device, epoch, step):
    vae.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    count = 0
    criterion = torch.nn.MSELoss()
    # 标记：是否已经对第一个 batch 进行可视化
    first_batch_visualized = False
    with torch.no_grad():
        for i, imgs in enumerate(val_loader):
            imgs = imgs.to(device, non_blocking=True)
            # ---- 对 imgs 做零填充 (F.pad) 以保证宽高能被 8 整除 ----
            B, C, T, H, W = imgs.shape
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            if pad_h > 0 or pad_w > 0:
                imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='constant', value=0)
            # ----------------------------
            encode_out = vae.encode(imgs, return_dict=True)
            posterior = encode_out.latent_dist
            kl_loss = posterior.kl().mean()
            # reparameterize: 这里采用随机采样
            z = posterior.sample()
            decode_out = vae.decode(z, return_dict=True)
            recons = decode_out.sample
            # 针对当前任务：如果 recons 的通道数为1 而 imgs 通道为3，则重复扩展
            if recons.shape[2] == 1 and imgs.shape[1] == 3:
                recons = recons.repeat(1, 1, 3, 1, 1)
            recon_loss = criterion(recons, imgs)
            loss = recon_loss + kl_loss
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            count += 1

            # 对第一个 batch 第一个样本进行可视化
            if not first_batch_visualized:
                # 调用辅助函数进行可视化
                vis_prefix = os.path.join(args.validate_path, f"val_epoch{epoch}_step{step}")
                visualize_sample(imgs[0], recons[0], vis_prefix, args.fps)
                first_batch_visualized = True

            if i >= args.max_val_steps - 1:
                break
    avg_loss = total_loss / count
    avg_recon = total_recon / count
    avg_kl = total_kl / count
    print(f"[Val] Epoch {epoch} Step {step}: Avg Loss: {avg_loss:.6f}, Recon Loss: {avg_recon:.6f}, KL Loss: {avg_kl:.6f}")
    vae.train()


def finetune_vae(args, vae, train_loader, val_loader, device):
    # 若使用分布式训练，先取出原始模型
    vae = vae.module if isinstance(vae, torch.nn.parallel.DistributedDataParallel) else vae
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = 500  # 500 warmup steps

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
        start_epoch = load_checkpoint(args.resume_ckpt, vae, optimizer, lr_scheduler)

    vae.train()

    # 初始化混合精度的 GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # 定义梯度累积步数（请在运行时通过 args 传入该参数，比如 --gradient_accumulation_steps 设为 4）
    accumulation_steps = args.gradient_accumulation_steps

    # 主训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        # 若使用了分布式Sampler，需要设置当前epoch，以便它做好shuffle
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # 清空梯度（开始新的梯度累积周期）
        optimizer.zero_grad()

        for step, imgs in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)

            # =====================
            # 1) 3DVAE前向传播（添加零填充保证宽高能被8整除）
            B, C, T, H, W = imgs.shape  # 例如：torch.Size([B, 3, 9, 300, 450])
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            if pad_h > 0 or pad_w > 0:
                imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='constant', value=0)

            with torch.cuda.amp.autocast():
                # (1) encode => 得到后验分布
                encode_out = vae.encode(imgs, return_dict=True) 
                posterior = encode_out.latent_dist
                kl_loss = posterior.kl().mean()  # KL 散度损失

                # (2) reparameterize (sampling)
                z = posterior.sample() # torch.Size([1, 16, 3, 38, 57])

                # (3) decode => 重构输出
                decode_out = vae.decode(z, return_dict=True)
                recons = decode_out.sample  # 例如：[B, 3, 9, 304, 456]
                # 若重构输出通道不匹配（例如 recons 为1通道，而 imgs 为3通道），则扩展通道

                # 计算重构损失（简单采用MSELoss）与总 loss
                recon_loss = F.mse_loss(recons, imgs)
                # total_loss = recon_loss + kl_loss
                total_loss = recon_loss

            # -----------------------------
            # 2) 梯度累积：对 loss 进行缩放后反向传播
            # -----------------------------
            # 这里将当前梯度贡献均摊到每个累积步中
            loss_to_accumulate = total_loss / accumulation_steps
            scaler.scale(loss_to_accumulate).backward()

            # 可选：打印日志
            if step % args.log_interval == 0:
                current_lr = lr_scheduler.get_lr() if lr_scheduler else args.lr
                print(f"[Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} "
                      f"ReconLoss: {recon_loss.item():.6f} "
                      f"KL: {kl_loss.item():.6f} "
                      f"TotalLoss: {total_loss.item():.6f} "
                      f"LR: {current_lr:.6f}")

            # 在第一个 epoch 第一个 step 时进行可视化（保存视频和9张图片）
            if epoch == start_epoch and step == 1:
                vis_prefix = os.path.join(args.checkpoint_dir, f"train_epoch{epoch}_step{step}")
                visualize_sample(imgs[0], recons[0], vis_prefix, args.fps)

            # # 中途进行验证
            # if step % args.val_interval == 0 and val_loader is not None:
            #     validate_vae(args, vae, val_loader, device, epoch, step)

            # # 定期保存 checkpoint
            # if step % args.save_interval == 0:
            #     save_path = os.path.join(args.checkpoint_dir, f"vqvae_epoch{epoch}_step{step}.pth")
            #     save_checkpoint(save_path, epoch, vae, optimizer, lr_scheduler)

            # 每 accumulation_steps 或当处于当前 epoch 最后一个 batch 时，更新参数
            if (step % accumulation_steps == 0) or (step == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if lr_scheduler:
                    lr_scheduler.step()

        # Epoch 结束时可选择保存 checkpoint
        if args.save_end_of_epoch:
            save_path = os.path.join(args.checkpoint_dir, f"vqvae_epoch{epoch}.pth")
            save_checkpoint(save_path, epoch, vae, optimizer, lr_scheduler)
            validate_vae(args, vae, val_loader, device, epoch, step)

    print("Training Finished!")

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
    parser.add_argument("--pkl_path", type=str, default="/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_data_dynamic_bottom_scaleddepth.pkl", help="Dataset name for logging")
    parser.add_argument("--validate_path", type=str, default='./validation/d408_1024', help="12")
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
    parser.add_argument("--input_height", type=int, default=520,
                        help="图像输入高度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients.")
    parser.add_argument("--input_width", type=int, default=784,
                        help="图像输入宽度")
    parser.add_argument("--inp_channels", type=int, default=80, help="输入通道数")
    parser.add_argument("--out_channels", type=int, default=80, help="输出通道数")
    parser.add_argument("--mid_channels", type=int, default=320, help="隐藏层通道数") # 1024
    parser.add_argument("--z_channels", type=int, default=4, help="潜变量通道数") # 256
    parser.add_argument("--fps", type=int, default=2, help="潜变量通道数") 
    parser.add_argument('--concat', action='store_true', help='...')
    # parser.add_argument("--dtype", type=str, default="float16",
    #                     help="The data type for computation (e.g., 'float16' or 'bfloat16')")

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

    dataset_train = MyDatasetforVAE3_3(args) 
    # dataset_train = MyDatasetforVAE_FRONT_1_9(args)
    
    num_tasks = distributed_mode.get_world_size() # 1
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
        pin_memory=args.pin_mem, # 当设置为 True 时，将数据加载到锁页内存中，这样可以加速数据从 CPU 到 GPU 的拷贝，尤其当你的 batch_size 较大时效果更明显
        drop_last=True, # 如果你的损失函数或模型对 batch_size 很敏感，设置 drop_last=True 可以避免最后一个 batch 尺寸不均的问题
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
            # 当数据样本的结构比较复杂或者每个样本大小不一致时，内置的 default_collate 可能无法满足需求
            # collate_fn=my_collate_fn, #! collate_fn 的职责是将多个样本组合成一个 batch
        )
        logger.info("Initialized a small validation set from training data")
    else:
        val_loader = None


    # todo：加载权重，路径：/mnt/bn/occupancy3d/workspace/mzj/CogVideoX-2b/vae/ 这个路径下有diffusion_pytorch_model.safetensors和config.json两个文件
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "/mnt/bn/occupancy3d/workspace/mzj/CogVideoX-2b/vae/",
        # torch_dtype= torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    )

    vae.to(device)

    if distributed_mode.get_world_size() > 1:
        vae = DDP(vae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # 根据warning修正

    vae_total_params = sum(p.numel() for p in vae.parameters())
    vae_trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"[Model Param Count] VQ-VAE total: {vae_total_params}, trainable: {vae_trainable_params}")
    
    finetune_vae(args, vae, train_loader, val_loader, device)

    print("Done!")


if __name__ == "__main__":
    main()