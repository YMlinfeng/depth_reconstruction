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
from tools.load_config import load_config
from torch.distributed.elastic.multiprocessing.errors import record

# 确保异常信息能够写入指定文件
if "TORCHELASTIC_ERROR_FILE" not in os.environ:
    # 这里将错误信息写入当前工作目录下的 error_log.json
    os.environ["TORCHELASTIC_ERROR_FILE"] = os.path.join(os.getcwd(), "error_log.json")

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


def train_vqvae(args, model, vqvae, train_loader, val_loader, device):
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
    model.eval()

    # 主训练循环，从 start_epoch 到 args.epochs，每个epoch遍历一遍 train_loader
    for epoch in range(start_epoch, args.epochs + 1):
        # 若使用了分布式Sampler，需要设置当前epoch，以便它做好shuffle
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for step, (imgs, img_metas) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)       
            with torch.no_grad():
                voxel3 = model(imgs, img_metas)  # voxel[0].shape: [B, 4, 60, 100, 20]
            voxel = voxel3[1]
            # vqvae_out = vqvae(voxel)
            # logits = vqvae_out['logits']
            # print("logits.is_leaf =", logits.is_leaf)
            # logits.retain_grad()  # ✅ 显式保留中间张量的梯度
            # # logits.requires_grad_(True)

            # # 只保留一小段 Decode Path
            # embed_loss = vqvae_out['embed_loss']     # 量化损失(标量)
            # loss = F.mse_loss(logits, torch.zeros_like(logits)) + embed_loss
    
            # depths, rgbs = model.module.pts_bbox_head.decode_sdf([voxel, logits], img_metas)

            # dloss = F.l1_loss(depths[0], depths[1]) + F.l1_loss(rgbs[0], rgbs[1])
            # loss += dloss
            # loss.backward()

            # print(logits.grad is not None)  # True → 表示 loss 能回传到 logits，计算图是通的

            # print("++++++++++++++++++")

            vqvae_out = vqvae(voxel)   # 包含 'logits', 'embed_loss'
            reconstructed_sdf = vqvae_out['logits']  # 重构结果 [B,4,60,100,20]
            reconstructed_sdf.retain_grad()
            embed_loss = vqvae_out['embed_loss']     # 量化损失(标量)
            quant = vqvae_out['quant']
            input_tensor = vqvae_out['input']
            quant.retain_grad()

            core = model.module if hasattr(model, 'module') else model
            depths, rgbs = core.pts_bbox_head.decode_sdf([input_tensor, reconstructed_sdf], img_metas)
            recon_loss = F.l1_loss(depths[0], depths[1]) + F.l1_loss(rgbs[0], rgbs[1])
          
            # 加一个 dummy loss，确保所有模块输出都参与 loss
            dummy = 0.0
            for name, param in vqvae.named_parameters():
                # if 'gpt' in name:
                dummy = dummy + 0.0 * (param ** 2).sum()

            total_loss = recon_loss + embed_loss + dummy  
            
            optimizer.zero_grad()
            total_loss.backward()


            if step == 1:
                # from torchviz import make_dot
                # make_dot(total_loss, params=dict(vqvae.named_parameters())).render("graph", format="pdf")
                # print("✅ 计算图生成完成，保存在当前目录的 graph.pdf")
                no_grad_params = []
                for name, param in vqvae.named_parameters():
                    if param.requires_grad and param.grad is None:
                        no_grad_params.append(name)

                if len(no_grad_params) == 0:
                    print("✅ All parameters have gradients.")
                else:
                    for name in no_grad_params:
                        print(f"[No Grad] {name}")

            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_lr()
            else:
                current_lr = args.lr
                
            if step % args.log_interval == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} "
                      f"ReconLoss: {recon_loss.item():.8f}, EmbedLoss: {embed_loss.item():.4f}, TotalLoss: {total_loss.item():.4f}, "
                      f"Current LR: {current_lr:.6f}")
            
            # # 中途也可以做一次简单验证
            # if step % args.val_interval == 0 and val_loader is not None:
            #     if args.general_mode == "vqgan":
            #         validate_vqvae(args, vqvae, val_loader, model, device, epoch, step)
            #     else:
            #         validate_vae(args, vqvae, val_loader, device, epoch, step)
            
            if step % args.save_interval == 0:
                save_path = os.path.join(args.checkpoint_dir, f"{args.general_mode}_epoch{epoch}_step{step}.pth")
                save_checkpoint(save_path, epoch, vqvae, optimizer, lr_scheduler)

        if args.save_end_of_epoch:
            save_path = os.path.join(args.checkpoint_dir, f"{args.general_mode}_epoch{epoch}.pth")
            save_checkpoint(save_path, epoch, vqvae, optimizer, lr_scheduler)
    
    print("Training Finished!")

@record
def main():

    parser = argparse.ArgumentParser("Train VQ-VAE/VAE Model")
    # ---------------------------
    # 分布式训练参数
    # ---------------------------
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
    parser.add_argument("--general_mode", type=str, default="vae", help="choices: vae, vqgan")
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
    parser.add_argument("--input_height", type=int, default=518,help="图像输入高度")
    parser.add_argument("--input_width", type=int, default=784,help="图像输入宽度")
    parser.add_argument("--inp_channels", type=int, default=80, help="输入通道数")
    parser.add_argument("--out_channels", type=int, default=80, help="输出通道数")
    parser.add_argument("--mid_channels", type=int, default=320, help="隐藏层通道数") # 1024
    parser.add_argument("--z_channels", type=int, default=4, help="潜变量通道数") # 256
    parser.add_argument("--img_shape", type=lambda s: tuple(map(int, s.split(','))), default="80,60,100,1", help="图像数据形状")
    parser.add_argument("--encoder_type", type=str, default="vqgan", help="Encoder类型") # 可选vqgan or vqgan_lc   
    parser.add_argument("--vq_config_path", type=str, default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/vqganlc/vqgan_configs/vqganlc_16384.yaml", help="Encoder类型")    
    parser.add_argument("--tuning_codebook", type=int, default=-1, help="Frozen or Tuning Coebook")
    parser.add_argument("--use_cblinear", type=int, default=2, help="Using Projector") # 1是Linear，2是MLP
    parser.add_argument("--quantizer_type", type=str, default="default", help="Quantizer类型") # 非 EMA 情况
    # parser.add_argument("--local_embedding_path", default="cluster_codebook_1000cls_100000.pth")
    parser.add_argument("--n_vision_words", type=int, default=16384, help="Codebook Size")
    parser.add_argument("--e_dim", type=int, default=1600, help="Codebook Size")
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

    # =======================
    # 1) 数据加载
    # =======================
    logger.info(f"Initializing Dataset: {args.dataset}")
    if args.general_mode == "vae":
        dataset_train = MyDatasetOnlyforVAE(args) 
    elif args.general_mode == "vqgan":
        dataset_train = MyDatasetOnlyforVoxel(args)
    # 获取总进程(卡)数
    num_tasks = distributed_mode.get_world_size() # 1
    # 获取当前进程（卡）编号
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

    if args.general_mode == "vqgan":
        model = LSSTPVDAv2OnlyForVoxel(num_classes=4, args=args)
        print("模型初始化完毕")
        state_dict = torch.load(
            '/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth',
            map_location='cpu'
        )['state_dict']
        model.load_state_dict(state_dict, strict=False)
        # print(model.load_state_dict(state_dict, strict=False))
        print("加载原始权重文件完毕") # 仅占用1492MB显存
        # for name, param in model.named_parameters():
        #     param.requires_grad = False
        # for param in model.parameters():
        #     param.requires_grad = False  # 本模型仅用于推理，不参与梯度更新
        model.to(device) 
        model.eval()

    # =======================
    # 3) 构造 VQ-VAE (训练模型)
    # =======================
    # 注意：已经有定义，这里只需要实例化即可
    vqvae_cfg = None  # 如果有额外vqvae的配置，可在此传

    if args.model == "VAERes2DImgDirectBC":
        vqvae = VAERes2DImgDirectBC(
            inp_channels=args.inp_channels,
            out_channels=args.out_channels,
            mid_channels=args.mid_channels,
            z_channels=args.z_channels,
            vqvae_cfg=vqvae_cfg
        )
    elif args.model == "VAERes3DImgDirectBC":
        vqvae = VAERes3DImgDirectBC(
            args=args,
            inp_channels=args.inp_channels,
            out_channels=args.out_channels,
            mid_channels=args.mid_channels,
            z_channels=args.z_channels,
            vqvae_cfg=vqvae_cfg
        )
    elif args.model == "VQModel":
        config = load_config(args.vq_config_path, display=True)
        vqvae = VQModel(args=args, ddconfig=config.model.params.ddconfig)
    elif args.model == "Cog3DVAE":
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
    # -------------------------------------------------------------------
    # 以下为多机多卡支持：若world_size>1，就用DDP封装 model 和 vqvae
    # 注意：LSSTPVDAv2OnlyForVoxel 只推理，不训练；若仍想让其多卡并行以分摊显存，也可DDP
    # -------------------------------------------------------------------
    if distributed_mode.get_world_size() > 1:
        if args.general_mode == "vqgan":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        vqvae = DDP(vqvae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) # 根据warning修正

    vqvae_total_params = sum(p.numel() for p in vqvae.parameters())
    vqvae_trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
    # print(f"[Model Param Count] LSSTPVDAv2OnlyForVoxel total: {model_total_params}, trainable: {model_trainable_params}")
    print(f"[Model Param Count] VQ-VAE total: {vqvae_total_params}, trainable: {vqvae_trainable_params}")
    
    try:
        train_vqvae(args, model, vqvae, train_loader, val_loader, device)
    except Exception as e:
        # 如果发生错误，这里可以打印日志，@record 会自动捕获异常并将详细错误信息写入 TORCHELASTIC_ERROR_FILE 指定的文件中
        print("训练过程中出现异常:", e)
        raise

    print("Done!")


if __name__ == "__main__":
    main()