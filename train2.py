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

# ===== 自定义模块 =====
from vqvae.vae_2d_resnet import VAERes2DImg, VAERes2DImgDirectBC, VectorQuantizer
from vqganlc.models.models_vq import VQModel
from training.pretrain_dataset import MyDatasetOnlyforVoxel
from training.train_arg_parser import get_args_parser
from model import LSSTPVDAv2OnlyForVoxel, LSSTPVDAv2
from vismask.gen_vis_mask import sdf2occ
from training import distributed_mode
from training.schedulers import WarmupCosineLRScheduler, SchedulerFactory


logger = logging.getLogger(__name__)

def my_collate_fn(batch):
    """
    batch 是一个列表，每个元素形如 (imgs_tensor, [img_meta])
    我们对 imgs_tensor 使用 torch.stack 进行堆叠，
    而对元数据则直接提取（注意这里假设每个样本返回的是一个只有一个 dict 的 list）。
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    # 由于 __getitem__ 返回的是 [img_meta]，因此取出列表内的第一个元素
    metas = [item[1][0] for item in batch]
    return imgs, metas

# ==============================
# save_checkpoint: 用于将当前模型和优化器的状态保存到本地文件，
# 以实现断点续训或事后analysis。
# ==============================
def save_checkpoint(save_path, epoch, vqvae, optimizer, lr_scheduler):
    # 准备保存字典，包含模型参数（vqvae.state_dict())、优化器参数、lr调度器参数、当前epoch等信息
    checkpoint = {
        'epoch': epoch,
        'vqvae_state_dict': vqvae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
    }
    torch.save(checkpoint, save_path)
    print(f"=> Saved checkpoint to {save_path}")


# ==============================
# load_checkpoint: 在需要断点续训时，从已有的 checkpoint 文件中恢复
# 训练的进度、模型参数、优化器参数等。
# ==============================
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


# ==============================
# validate_vqvae: 这里定义一个简单的验证过程，对一小部分batch进行推理，
# 统计重构损失和embed_loss，帮助监控模型收敛情况。
# ==============================
def validate_vqvae(args, vqvae, val_loader, model, device, epoch, step):
    """
      1. 每次验证时，会创建一个新的子目录，目录名称包含当前 epoch 和 step 信息
      2. 验证过程中所有的 loss 信息会写入该目录下的日志文件中
    """
    vqvae.eval()   # 关闭 dropout / batchnorm 动态统计
    total_loss = 0
    total_steps = 0

    # 构造保存验证结果的文件夹，子目录名称包含 epoch 和 step 信息
    visual_dir = os.path.join(args.validate_path, f'epoch_{epoch}_step_{step}')
    os.makedirs(visual_dir, exist_ok=True)

    # 用于存储每个batch的日志信息
    log_lines = []

    with torch.no_grad():
        for imgs, img_metas in val_loader:
            imgs = imgs.to(device)
            # 通过已训练好的推理模型获得 voxel 信息
            result3 = model(imgs, img_metas)  # 例如: [B, 4, 60, 100, 20]
            # 取第一个输出作为 voxel
            result = result3[0]
            # VQ-VAE前向传播
            vqvae_out = vqvae(result)
            reconstructed_sdf = vqvae_out['logits']    # [B, 4, 60, 100, 20]
            embed_loss = vqvae_out['embed_loss']

            # 计算重构损失
            recon_loss = F.mse_loss(reconstructed_sdf, result)
            batch_loss = recon_loss + embed_loss
            total_loss += batch_loss.item()
            total_steps += 1

            # 保存损失信息到日志
            log_lines.append(f"Batch {total_steps}: ReconLoss = {recon_loss.item():.6f}, EmbedLoss = {embed_loss.item():.6f}, TotalLoss = {batch_loss.item():.6f}\n")

            # 保存验证过程中的图像
            views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
            input_shape = [518, 784]
            # 假设 result3[-1] 包含的是深度信息，重新调整张量形状后保存图像
            depth = result3[-1].view(len(views), input_shape[0] // 2, input_shape[1] // 2).detach().squeeze().cpu().numpy()
            for i in range(len(views)):
                depth_img = np.log10(depth[i] + 1e-8)  # 加上一个很小的数，避免 log(0)
                plt.imsave(os.path.join(visual_dir, f'depth_{i}.png'), depth_img, cmap='jet')

            # 若只需要验证极少量的batch，则提前退出
            if total_steps >= args.max_val_steps:
                break

    avg_loss = total_loss / max(1, total_steps)
    log_lines.append(f"\n[Validation] Average Loss = {avg_loss:.6f}\n")

    # 将日志写入到指定文件中
    log_file_path = os.path.join(visual_dir, "validation_log.txt")
    with open(log_file_path, "w") as f:
        f.writelines(log_lines)

    print(f"[Validation] avg loss = {avg_loss:.6f} (log and images saved to: {visual_dir})")
    vqvae.train()  # 切换回训练模式
    return avg_loss


# ==============================
# train_vqvae: 这是核心训练循环。
# 1) 取数据：从 train_loader 取 imgs, img_metas
# 2) 不计算梯度地通过 model(LSSTPVDAv2OnlyForVoxel) 获得 voxel
# 3) voxel 送入 vqvae，得到重构后的 sdf 和 embed_loss
# 4) 计算总 loss = recon_loss + embed_loss
# 5) backward + optimizer.step 更新参数
# 6) 定期保存 checkpoint（断点续训）
# 7) 定期进行验证 validate_vqvae()
# ==============================
def train_vqvae(args, model, vqvae, train_loader, val_loader, device):
    # optimizer = torch.optim.Adam(vqvae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer = torch.optim.AdamW(vqvae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # # 如果需要学习率调度，可以创建

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

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
    model.eval()

    # 在训练开始之前，打印一下当前 GPU 显存占用
    print(f"[Before Training] GPU Memory allocated: {torch.cuda.memory_allocated(device)/1024/1024:.2f} MB | " \
          f"max allocated: {torch.cuda.max_memory_allocated(device)/1024/1024:.2f} MB")

    # 主训练循环，从 start_epoch 到 args.epochs，每个epoch遍历一遍 train_loader
    for epoch in range(start_epoch, args.epochs + 1):
        # 若使用了分布式Sampler，需要设置当前epoch，以便它做好shuffle
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for step, (imgs, img_metas) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            
            # ===================== 
            # 1) 得到 voxel (B,4,60,100,20) 
            # =====================
            with torch.no_grad():
                # 不需要计算梯度，让网络保持eval
                voxel3 = model(imgs, img_metas)  # voxel[0].shape: [B, 4, 60, 100, 20]

            # =====================
            # 2) VQ-VAE前向传播
            # =====================
            # import pdb; pdb.set_trace()
            voxel = voxel3[0]
            vqvae_out = vqvae(voxel)   # 包含 'logits', 'embed_loss', 'mid'
            reconstructed_sdf = vqvae_out['logits']  # 重构结果 [B,4,60,100,20]
            embed_loss = vqvae_out['embed_loss']     # 量化损失(标量)
            
            # =====================
            # 3) 计算loss并反向传播
            # =====================
            recon_loss = F.mse_loss(reconstructed_sdf, voxel)  # 与输入 voxel 做 MSE
            total_loss = recon_loss + embed_loss               # 总损失
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_lr()
                
            # 打印或记录训练日志
            if step % args.log_interval == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} "
                      f"ReconLoss: {recon_loss.item():.8f}, EmbedLoss: {embed_loss.item():.4f}, TotalLoss: {total_loss.item():.4f}, "
                      f"Current LR: {current_lr:.6f}")
            
            # 中途也可以做一次简单验证
            if step % args.val_interval == 0 and val_loader is not None:
                validate_vqvae(args, vqvae, val_loader, model, device, epoch, step)
            
            # =====================
            # 4) 定期保存checkpoint
            # =====================
            if step % args.save_interval == 0:
                save_path = os.path.join(args.checkpoint_dir, f"vqvae_epoch{epoch}_step{step}.pth")
                save_checkpoint(save_path, epoch, vqvae, optimizer, lr_scheduler)
        
        # 每个epoch结束后也可以保存一次
        if args.save_end_of_epoch:
            save_path = os.path.join(args.checkpoint_dir, f"vqvae_epoch{epoch}.pth")
            save_checkpoint(save_path, epoch, vqvae, optimizer, lr_scheduler)
    
    print("Training Finished!")


# ==============================
# main: 主函数，在这里：
#   1) 设置参数
#   2) 构建DataLoader
#   3) 初始化LSSTPVDAv2OnlyForVoxel并加载预训练权重
#   4) 初始化VQ-VAE
#   5) 调用 train_vqvae()来训vqvae
# ==============================
def main():

    parser = argparse.ArgumentParser("Train VAERes2DImg VQ-VAE Model")
    # ---------------------------
    # 分布式训练参数
    # ---------------------------
    parser.add_argument("--dist_on_itp",
                        action="store_true",
                        default=False,
                        help="使用基于 ITp 的分布式模式（如 MPI 的 OMPI 环境） if set.")
    # parser.add_argument("--dist_url",
    #                     type=str,
    #                     default="tcp://10.124.2.134:12355",
    #                     help="用于分布式训练初始化的 URL (例如: tcp://主节点IP:端口).")
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
    # ==============================
    # 这里是一些与训练VQ-VAE相关的参数,可随意添加
    # ==============================
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
    parser.add_argument("--save_end_of_epoch", action='store_true', help="save checkpoint at each epoch end") #flag
    parser.add_argument("--use_scheduler", action='store_false', help="use StepLR scheduler or not")
    parser.add_argument("--max_val_steps", type=int, default=5, help="max batch for val step")

    # 以下是模型参数，与VQ-VAE定义对应（示例）
    parser.add_argument("--inp_channels", type=int, default=80, help="输入通道数")
    parser.add_argument("--out_channels", type=int, default=80, help="输出通道数")
    parser.add_argument("--mid_channels", type=int, default=320, help="隐藏层通道数") # 1024
    parser.add_argument("--z_channels", type=int, default=4, help="潜变量通道数") # 256
    parser.add_argument("--img_shape", type=lambda s: tuple(map(int, s.split(','))), default="80,60,100,1", help="图像数据形状")
    

    args = parser.parse_args()

    # -------------------------------
    # 初始化分布式环境 (多机多卡 or 单机多卡)
    # -------------------------------
    distributed_mode.init_distributed_mode(args)
    local_rank = args.gpu
    torch.cuda.set_device(local_rank)
    
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # =======================
    # 1) 数据加载
    # =======================
    logger.info(f"Initializing Dataset: {args.dataset}")
    dataset_train = MyDatasetOnlyforVoxel() 
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
        collate_fn=my_collate_fn
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
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            collate_fn=my_collate_fn
        )
        logger.info("Initialized a small validation set from training data")

    # =======================
    # 2) 构造LSSTPVDAv2OnlyForVoxel模型 (推理模型, 不参与训练)
    # =======================
    model = LSSTPVDAv2OnlyForVoxel(num_classes=4)
    print("模型初始化完毕")
    state_dict = torch.load(
        '/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth',
        map_location='cpu'
    )['state_dict']
    # model.load_state_dict(state_dict, strict=False)
    print(model.load_state_dict(state_dict, strict=False))
    print("加载原始权重文件完毕") # 仅占用1492MB显存
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False  # 本模型仅用于推理，不参与梯度更新
    model.to(device) 
    model.eval()

    # =======================
    # 3) 构造 VQ-VAE (训练模型)
    # =======================
    # 注意：已经有 VAERes2DImg 的定义，这里只需要实例化即可
    vqvae_cfg = None  # 如果有额外vqvae的配置，可在此传
    # vqvae = VAERes2DImg(
    vqvae = VAERes2DImgDirectBC(
        inp_channels=args.inp_channels,
        out_channels=args.out_channels,
        mid_channels=args.mid_channels,
        z_channels=args.z_channels,
        vqvae_cfg=vqvae_cfg
    )
    vqvae.to(device)

    # -------------------------------------------------------------------
    # 以下为多机多卡支持：若world_size>1，就用DDP封装 model 和 vqvae
    # 注意：LSSTPVDAv2OnlyForVoxel 只推理，不训练；若仍想让其多卡并行以分摊显存，也可DDP
    # -------------------------------------------------------------------
    if distributed_mode.get_world_size() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        vqvae = DDP(vqvae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) # 根据warning修正
        print("[DDP] Wrapped both model and vqvae with DistributedDataParallel")

    # =======================
    # 统计两模型的参数量，并打印显存占用
    # =======================
    # 对 model (LSSTPVDAv2OnlyForVoxel) 进行统计
    model_total_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 对 vqvae 进行统计
    vqvae_total_params = sum(p.numel() for p in vqvae.parameters())
    vqvae_trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
    print(f"[Model Param Count] LSSTPVDAv2OnlyForVoxel total: {model_total_params}, trainable: {model_trainable_params}")
    print(f"[Model Param Count] VQ-VAE total: {vqvae_total_params}, trainable: {vqvae_trainable_params}")

    current_alloc = torch.cuda.memory_allocated(device)/1024/1024
    current_max_alloc = torch.cuda.max_memory_allocated(device)/1024/1024
    print(f"[After Model Init] GPU Memory usage allocated: {current_alloc:.2f} MB, max allocated: {current_max_alloc:.2f} MB")

    # =======================
    # 4) 进入训练循环
    # =======================
    train_vqvae(args, model, vqvae, train_loader, val_loader, device)

    # 结束
    print("Done!")


if __name__ == "__main__":
    # import debugpy
    # # 监听端口
    # debugpy.listen(("127.0.0.1", 5679))
    # # 等待调试器连接（可选）
    # print("等待调试器连接...") #按F5
    # debugpy.wait_for_client()


    main()