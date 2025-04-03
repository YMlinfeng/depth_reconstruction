#!/usr/bin/env python3
import argparse
import os
import math
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from vqvae.vae_2d_resnet import VAERes2DImg
from vqvae.vae_2d_resnet import VectorQuantizer
# 假设你的模型文件和辅助模块已经在相应文件夹中
# 如果你的模型放在 model.py 中，则如下导入
from training.pretrain_dataset import  MyDatasetOnlyforVoxel
from training.train_arg_parser import get_args_parser
from model import LSSTPVDAv2OnlyForVoxel
from vismask.gen_vis_mask import sdf2occ
from training import distributed_mode
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import math
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from vqvae.vae_2d_resnet import VAERes2DImg
from vqvae.vae_2d_resnet import VectorQuantizer
from training.pretrain_dataset import MyDataset
from training.train_arg_parser import get_args_parser
from model import LSSTPVDAv2OnlyForVoxel
from vismask.gen_vis_mask import sdf2occ
from training import distributed_mode
import matplotlib.pyplot as plt
from model import LSSTPVDAv2
import pickle
import numpy as np
import cv2
import torch.nn.functional as F

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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("Train VAERes2DImg VQ-VAE Model")
    # 模型相关参数（可根据需要调整）
    parser.add_argument("--inp_channels", type=int, default=80, help="输入图像通道数") # 3
    parser.add_argument("--out_channels", type=int, default=80, help="输出图像通道数") # 3
    parser.add_argument("--mid_channels", type=int, default=320, help="中间表示通道数") # 256
    parser.add_argument("--z_channels", type=int, default=4, help="潜变量 z 的通道数")
    # 图像数据形状：格式为 "c,h,w,d"，例如 "80,60,100,1"
    parser.add_argument("--img_shape", type=lambda s: tuple(map(int, s.split(','))), default="80,60,100,1", help="图像数据形状")

    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ==================== 数据集和数据加载 ====================
    logger.info(f"Initializing Dataset: {args.dataset}")
    dataset_train = MyDatasetOnlyforVoxel() 
    # 获取总进程数
    num_tasks = distributed_mode.get_world_size()
    # 获取当前进程编号
    global_rank = distributed_mode.get_rank()   
    # 构造分布式采样器
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True
    )        
    # 构造 DataLoader
    train_loader = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,   # 每个进程上的 batch_size
        num_workers=args.num_workers, # 读数据的线程数
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=my_collate_fn       # 使用自定义 collate_fn
    )
    logger.info("Intializing DataLoader")

    '''
    dataset_train.data[0]
    {'frame_idx': 120, 'timestamp': 120, 'scene_token': '2024_06_26-21_43_13-client_81567f59-dd29-4deb-82d2-b5507cceb3b2', 'cams': {'CAM_FRONT': {...}, 'CAM_FRONT_LEFT': {...}, 'CAM_FRONT_RIGHT': {...}}}
    special variables
    function variables
    'frame_idx' =
    120
    'timestamp' =
    120
    'scene_token' =
    '2024_06_26-21_43_13-client_81567f59-dd29-4deb-82d2-b5507cceb3b2'
    'cams' =
    {'CAM_FRONT': {'data_path': 'data/dcr_data_pretrain3d/2024_06_26-21_43_13-client_81567f59-dd29-4deb-82d2-b5507cceb...mages/bottom_front/1719409886260512512.jpg', 
                    'cam_intrinsic': array([[1.0175e+03, 0.0000e+00, 9.6800e+02],
        [0.0000e+00, 1.0180e+03, 5.8600e+...    [0.0000e+00, 0.0000e+00, 1.0000e+00]]), 
        'cam_extrinsic': array([[ 0.        ,  0.20788574,  0.97802734,  0.40307617],
        [-1.        , -0....,  0.        ,  0.        ,  1.        ]]), 
        'depth_path': 'data/dcr_data_pretrain3d/2024_06_26-21_43_13-client_81567f59-dd29-4deb-82d2-b5507cceb...caled/bottom_front/1719409886260512512.png'}, 
    'CAM_FRONT_LEFT': {'data_path': 'data/dcr_data_pretrain3d/2024_06_26-21_43_13-client_81567f59-dd29-4deb-82d2-b5507cceb.../bottom_front_left/1719409886263462400.jpg', 'cam_intrinsic': array([[1.0145e+03, 0.0000e+00, 9.6150e+02],
        [0.0000e+00, 1.0145e+03, 6.0450e+...    [0.0000e+00, 0.0000e+00, 1.0000e+00]]), 'cam_extrinsic': array([[ 0.70898438,  0.1451416 ,  0.69042969,  0.36401367],
        [-0.70507812,  ...
    len() =4
    '''
    # 接下来是得到voxel（B，4，60，100，20）的过程
    model = LSSTPVDAv2OnlyForVoxel(num_classes=4)
    # 加载原始权重文件
    print("模型初始化完毕")
    state_dict = torch.load(
        '/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth',
        map_location='cpu'
    )['state_dict']
    model.load_state_dict(state_dict, strict=False)
    print("加载原始权重文件完毕")

    model.to(device) # 这行代码 将 model 从 CPU 内存移动到 GPU 显存（通常是 cuda:0）
    model.eval() # 关闭 Dropout（用于训练时随机丢弃部分神经元，使得推理结果稳定）；让 Batch Normalization（BN）层使用保存的均值和方差（而不是训练时的动态计算）；计算图 不会保存反向传播的梯度，减少显存占用
   


    # todo 以下模拟一个简单的训练循环，供你理解数据的预处理过程
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 保存路径
    visual_dir = './output/d401/t1'
    views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    input_shape = [518, 784]  # !目标输入分辨率
    for imgs, img_metas in train_loader:
        imgs = imgs.to(device)
        print("imgs shape:", imgs.shape)
        print("len(img_metas):", len(img_metas))
        print("img_metas keys:", img_metas[0].keys()) # # dict_keys(['pc_range', 'occ_size', 'lidar2img', 'lidar2cam', 'cam_intrinsic', 'img_shape'])
        # 推理模型
        with torch.no_grad():
            result = model(imgs, img_metas) # torch.Size([1, 4, 60, 100, 20]) # #todo：这就是vqvae的输入
        # todo 开始训练
        break  # 只看一次即可
    
    

if __name__ == "__main__":
    # import debugpy
    # # 监听端口
    # debugpy.listen(("127.0.0.1", 5678))
    # # 等待调试器连接（可选）
    # print("等待调试器连接...") #按F5
    # debugpy.wait_for_client()
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)