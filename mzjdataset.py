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

# 把原来的 get_data_path 修改为写死 DATA_ROOT 的版本
def get_data_path(relative_path=""):
    data_root = "/mnt/bn/occupancy3d/real_world_data/preprocess"
    return os.path.join(data_root, relative_path)
# ==============================
# 宏观解释：Dataset类用于定义数据集的读取、预处理等操作
# MyDatasetOnlyforVoxel 专门用来生成可供VQVAE训练的输入（voxel）。
# ==============================
class MyDatasetOnlyforVoxel(Dataset):
    """
    这个类的主要职责是：
    1. 读取事先准备好的数据文件（pickle 文件中存放了多个帧的数据），其中包含多帧图像及对应的传感器外参、内参信息。
    2. 通过 __getitem__，一次性返回一个包含多帧（比如4帧）有效数据的样本，包括：
       - 图像张量 (imgs)，已经完成归一化、减均值除标准差、以及尺寸插值。
       - 相机和自车坐标系之间的变换矩阵 (ego2imgs, ego2cams, cam_intrinsic)，以及其他元数据 (img_meta)。
    3. 下游在使用 DataLoader 迭代这个 Dataset 时，就能很方便地把 imgs, img_metas 输入到您的模型 model(imgs, img_metas)，
       得到 voxel，再进一步送进VQVAE进行训练。
    """

    def __init__(self):
        # ==============================
        # 宏观解释：在初始化中，我们需要做的事情：
        # 1. 设定多帧相关参数 (tem, frames)。
        # 2. 读取数据源pickle文件，把它加载到 self.data 里。
        # 3. 定义一些常量，比如图像归一化所需的 mean, std，以及网络的目标输入分辨率 input_shape。
        # ==============================
        
        # 时间步长：表示在数据中，采样下一个帧的间隔
        self.tem = 1  
        # 连续的帧数：一个样本包含多少个连续帧
        self.frames = 1  
        
        data_pkl = '/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_data_bottom_static_scaleddepth_240925_sub.pkl'
        self.data = pickle.load(open(data_pkl, 'rb'))['infos']
        
        # ImageNet常用的归一化均值和标准差
        # 形状 [1,1,3] 的原因是后续做广播运算时更方便
        self.mean = np.array([0.485, 0.456, 0.406])[None, None, :]  
        self.std = np.array([0.229, 0.224, 0.225])[None, None, :]  
        
        # 目标输入分辨率(高, 宽)，与 inference 时保持一致
        self.input_shape = [518, 784]  
        
        # 使用的相机视角列表，根据您之前的推理代码
        self.views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    
    def __len__(self):
        """
        返回数据集总帧数（并非总样本数，因为一条样本可能由4帧组成）。
        """
        return len(self.data)

    def __getitem__(self, idx):
        # ==============================
        # 宏观解释：在 __getitem__ 中，每次我们需要返回一个“多帧”样本（frames=1），
        # 保证取到相同scene下的连续帧。若 idx 所指位置不足以取到4帧，则往前回退。
        # ==============================
        
        # 如果无法满足取连续4帧(或者不同帧scene_token不一致)，则往前移一个tem*(frames-1)步
        # 以保证能在同一个scene里拿到满足frames个数的连续帧
        # if ((idx + self.tem * (self.frames - 1)) >= len(self)) or \
        #    (self.data[idx]['scene_token'] != self.data[idx + self.tem * (self.frames - 1)]['scene_token']):
        #     idx = idx - self.tem * (self.frames - 1)

        # # 确定我们要的4帧的具体索引
        # index_list = [idx + i * self.tem for i in range(self.frames)]
        
        # 当 frames=1 时 (self.frames - 1) = 0，以下判断其实很少会触发调整
        if ((idx + self.tem * (self.frames - 1)) >= len(self)) or \
           (self.data[idx]['scene_token'] != self.data[idx + self.tem * (self.frames - 1)]['scene_token']):
            idx = idx - self.tem * (self.frames - 1)

        # 只会得到 [idx] 一个索引
        index_list = [idx + i * self.tem for i in range(self.frames)]

        # ==============================
        # 宏观解释：接下来，我们会把这4帧中，每一帧涉及到的3个相机图像全部读进来，并做和您推理时
        # 一样的预处理，包括通道变换、归一化、尺寸插值、调整内参等。
        # 我们还会构造 img_metas，把 ego2imgs, ego2cams, cam_intrinsic 等都存进去。
        # ==============================
        
        # 用于存储所有帧的图像张量 (4帧 × 3视角)
        # 注意：最后我们会把它组织成一个 torch.Tensor 交给DataLoader
        all_imgs = []
        
        # 用于存储所有帧的投影矩阵、内外参等
        # 这里我们让它是4个元素的列表，每个元素对应当前帧下的“多相机信息”
        all_lidar2img = []   # shape: (frames, 3, 4, 4)
        all_lidar2cam = []   # shape: (frames, 3, 4, 4)
        all_cam_intrinsic = []  # shape: (frames, 3, 3, 3)

        # ==============================
        # 初始化一些常量，比如点云范围、体素维度等
        # 在原推理中，这些信息保存在 img_meta['pc_range']、img_meta['occ_size'] 中
        # 这里同样保留，以便下游网络使用。
        # ==============================
        pc_range = [0., -10., 0., 12., 10., 4.]
        occ_size = [60, 100, 20]
        
        # 遍历这4帧，在单帧情况下，该循环只跑1次，即 f_idx = idx
        for f_idx in index_list:
            item_info = self.data[f_idx]
            
            # 取出当前帧所在的 scene_token, timestamp 等信息（若您后续需要可保留）
            scene_token = item_info['scene_token']
            # timestamp = item_info['timestamp']  # 看需要是否要使用
            
            # 读取该帧下的相机图像及其内外参
            frame_imgs = []          # 保存该帧的3个相机图像(H, W, C)
            frame_lidar2img = []     # 保存该帧3个lidar2img(4x4)
            frame_lidar2cam = []     # 保存该帧3个lidar2cam(4x4)
            frame_cam_intrinsic = [] # 保存该帧3个内参(3x3)
            
            for view_id in self.views:
                cam_data = item_info['cams'][view_id]
                
                # 1) 读取图像
                img_path = cam_data['data_path']  # 图像路径
                # 如果路径以 /data 开头，则更换为绝对路径
                if img_path.startswith("data/"):
                    img_path = img_path.replace("data", "/mnt/bn/occupancy3d/real_world_data/preprocess", 1)
                img_bgr = cv2.imread(img_path)     # OpenCV读取到的通道顺序是BGR
                if img_bgr is None:
                    # 如果图像路径异常，简单地抛异常或返回全零图像(根据需求)
                    raise ValueError(f"Fail to read image: {img_path}")
                
                # 2) 转到RGB通道，并归一化到[0,1]
                img_rgb = img_bgr[:, :, [2, 1, 0]] / 255.0
                
                # 3) 减均值除标准差 (逐像素进行)
                img_normalized = (img_rgb - self.mean) / self.std
                
                # 注意：现阶段还未做resize (后面我们用torch的F.interpolate统一裁到 self.input_shape)
                frame_imgs.append(img_normalized)  # shape: (H, W, 3)
                
                # 读取相机的内外参
                extrinsic = cam_data['cam_extrinsic']  # shape: (4,4)
                intrinsic = cam_data['cam_intrinsic']  # shape: (3,3)
                
                # 先把intrinsic扩为4x4，以方便与extrinsic等相乘
                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                
                # 根据实际图像分辨率，构造 resize 的 scale_factor 矩阵
                src_H, src_W = img_bgr.shape[:2]
                scale_matrix = np.eye(4)
                scale_matrix[0, 0] = self.input_shape[1] / src_W   # 宽度缩放
                scale_matrix[1, 1] = self.input_shape[0] / src_H   # 高度缩放
                
                # ego2img = (scale * intrinsic) * inverse(extrinsic)
                # 用于从自车坐标(Ego)投影到图像坐标
                ego2img = scale_matrix @ viewpad @ np.linalg.inv(extrinsic)
                
                # ego2cam = inverse(extrinsic)
                # 用于从自车(Ego)转换到相机坐标
                ego2cam = np.linalg.inv(extrinsic)
                
                # 同时要更新一下 intrinsic 被resize 后的有效参数： intrinsic' = scaleMatrix[:3,:3] * intrinsic
                new_intrinsic = scale_matrix[:3, :3] @ intrinsic
                
                frame_lidar2img.append(ego2img)
                frame_lidar2cam.append(ego2cam)
                frame_cam_intrinsic.append(new_intrinsic)
            
            # end for view_id in self.views
            
            # 把该帧的3张图像拼成一个 np.array，以便后续转 torch
            # shape: (3, H, W, 3) —— 因为3个视角
            frame_imgs_np = np.stack(frame_imgs, axis=0)
            
            # 保存到all_imgs里
            all_imgs.append(frame_imgs_np)
            
            # 保存这帧对应的多个相机变换矩阵
            all_lidar2img.append(frame_lidar2img)
            all_lidar2cam.append(frame_lidar2cam)
            all_cam_intrinsic.append(frame_cam_intrinsic)

        # end for f_idx in index_list
        
        # ! 4 frames × 3 views
        # # ==============================
        # # 宏观解释：到这里，我们已经把4帧×3视角的图像，都以 numpy 的形式放进了 all_imgs (list)，
        # # 并且也记录了每帧对应的外参、内参（处理了 resize）信息。
        # # 接下来，我们需要：
        # # 1) 把 all_imgs 转成 torch.Tensor，
        # # 2) 用 F.interpolate 做统一的尺寸变换到 self.input_shape，
        # # 3) 把变换后的图像和变换矩阵们打包到 img_metas。
        # # ==============================
        
        # # 先把4帧的图像合并成一个维度大的 numpy 数组
        # # shape: (frames, 3, H, W, 3)
        # all_imgs_np = np.stack(all_imgs, axis=0)
        
        # # 转成 torch.Tensor，并把通道排列变成 (frames, 3, 3, H, W)
        # # 其中 3(第二维) 表示3个相机视角，另一个3(第三维)是RGB通道
        # # 所以最后维度解释为：
        # #   B = frames (4帧)
        # #   V = 3 (CAM_FRONT, etc)
        # #   C = 3 (颜色通道)
        # #   H, W = 原图像高宽
        # all_imgs_torch = torch.from_numpy(all_imgs_np).permute(0, 1, 4, 2, 3).float()
        
        # # 我们希望插值到 self.input_shape (高=518, 宽=784)
        # # 注意：all_imgs_torch 当前形状是 (4, 3, 3, H, W)
        # # 我们可以把前两个维度合并，做一次插值，再拆开。
        # B, V, C, H, W = all_imgs_torch.shape
        # all_imgs_torch = all_imgs_torch.reshape(B*V, C, H, W)  # -> (12, 3, H, W)
        
        # # 使用 PyTorch 提供的 F.interpolate 来做双线性插值
        # all_imgs_torch = F.interpolate(
        #     all_imgs_torch, 
        #     size=(self.input_shape[0], self.input_shape[1]),  # (H, W)
        #     mode='bilinear',
        #     align_corners=False
        # )
        
        # # 再把插值后的张量变回 (4, 3, 3, 518, 784)
        # all_imgs_torch = all_imgs_torch.reshape(B, V, C, self.input_shape[0], self.input_shape[1])
        
        # # ==============================
        # # 接下来为这个样本组织一个 img_meta，里面可以放您模型需要的各种额外信息，如
        # # pc_range, occ_size, lidar2img, lidar2cam, cam_intrinsic, etc.
        # # ==============================
        
        # # 我们这里示例：把4帧×3视角的矩阵都放 list 里即可
        # # 以下只是示例做法——实际中可以视网络需要拆分或拼接
        # img_meta = dict()
        # img_meta['pc_range'] = pc_range
        # img_meta['occ_size'] = occ_size
        # # 这里存的 lidars2img 是一个 (frames, 3, 4, 4) 的list(实际上是list of list of np.array)
        # img_meta['lidar2img'] = all_lidar2img
        # img_meta['lidar2cam'] = all_lidar2cam
        # img_meta['cam_intrinsic'] = all_cam_intrinsic
        
        # # img_shape 如果只需要存最终插值之后的分辨率，就存 [518, 784]
        # # 如果需要区分帧等，也可以存一个list
        # img_meta['img_shape'] = [self.input_shape[0], self.input_shape[1], 3]
        
        # # ==============================
        # # 最终返回值：
        # #   - all_imgs_torch: shape = (4, 3, 3, 518, 784)
        # #     表示4帧，每帧3个相机视角，每个相机图像是3通道(518x784)
        # #   - [img_meta]: 将其放入一个list中是因为推理时您示例代码中常用list包装
        # # ==============================
        # return all_imgs_torch, [img_meta]
        #! 4 frames × 3 views

        #! 1 frame 
        # 此时 all_imgs shape: (1, 3, H, W, 3)，因为只1帧、3视角
        all_imgs_np = np.stack(all_imgs, axis=0)  # (frames=1, 3, H, W, 3)
        
        # 转 torch, -> (1, 3, 3, H, W)
        all_imgs_torch = torch.from_numpy(all_imgs_np).permute(0, 1, 4, 2, 3).float()
        
        # 形状: 
        #   B=frames(1), V=3(视角), C=3(RGB), H=原高, W=原宽
        B, V, C, H, W = all_imgs_torch.shape
        all_imgs_torch = all_imgs_torch.reshape(B*V, C, H, W)
        
        # 插值到 self.input_shape
        all_imgs_torch = F.interpolate(
            all_imgs_torch, 
            size=(self.input_shape[0], self.input_shape[1]),
            mode='bilinear',
            align_corners=False
        )
        
        # 再 reshape 回 (3, 3, 518, 784)
        all_imgs_torch = all_imgs_torch.reshape(V, C, self.input_shape[0], self.input_shape[1])
        
        # 构造 img_meta
        pc_range = [0., -10., 0., 12., 10., 4.]
        occ_size = [60, 100, 20]
        
        img_meta = dict()
        img_meta['pc_range'] = pc_range
        img_meta['occ_size'] = occ_size
        img_meta['img_shape'] = [(self.input_shape[0], self.input_shape[1], 3)]
        if self.frames == 1:
            img_meta['lidar2img'] = all_lidar2img[0]  # (1, 3, 4, 4)
            img_meta['lidar2cam'] = all_lidar2cam[0]
            img_meta['cam_intrinsic'] = all_cam_intrinsic[0]
        
        return all_imgs_torch, [img_meta]

        #! 1 frame

# ==============================
# 宏观解释：有了 Dataset 之后，就可以使用 DataLoader 来并行加载、打包数据。
#   1. 分布式采样器 (DistributedSampler) 用于在多GPU/多进程情况下，每个GPU拿到数据集的一部分。
#   2. DataLoader 的 batch_size, num_workers 可以根据硬件资源设置。
#   3. 在训练循环中，就可以直接 for imgs, img_metas in data_loader_train: ... 来使用。
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
def build_dataloader_for_vqvae(args, distributed_mode):
    # 实例化自定义数据集
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
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,   # 每个进程上的 batch_size
        num_workers=args.num_workers, # 读数据的线程数
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=my_collate_fn       # 使用自定义 collate_fn
    )
    
    return data_loader_train


# ==============================
# 使用示例（假装我们有一个 args, distributed_mode）
# ==============================
if __name__ == "__main__":
    # import debugpy
    # # 监听端口
    # debugpy.listen(("127.0.0.1", 5678))
    # # 等待调试器连接（可选）
    # print("等待调试器连接...") #按F5
    # debugpy.wait_for_client()
    # 这里假设我们有简单的 args 和 distributed_mode 对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 保存路径
    visual_dir = './output/d401/'
    views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    input_shape = [518, 784]  # !目标输入分辨率
    # imgs = [cv2.imread(sample['images'][view]) for view in views]

    

    os.makedirs(visual_dir, exist_ok=True)
    class Args:
        batch_size = 1
        num_workers = 1
        pin_mem = True
    class DistMode:
        def get_world_size(self):
            return 1
        def get_rank(self):
            return 0
    
    args = Args()
    dist_mode = DistMode()
    
    # 构建 DataLoader
    train_loader = build_dataloader_for_vqvae(args, dist_mode)
    
    model = LSSTPVDAv2(num_classes=4)
    print("模型初始化完毕")
    # 下一步会 覆盖 model 之前的 随机初始化参数
    # 加载原始权重文件
    state_dict = torch.load(
        '/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth',
        map_location='cpu'
    )['state_dict']
    model.load_state_dict(state_dict, strict=False)
    print("加载原始权重文件完毕")

    # 预测occupancy
    model.to(device) # 这行代码 将 model 从 CPU 内存移动到 GPU 显存（通常是 cuda:0）
    model.eval() # 关闭 Dropout（用于训练时随机丢弃部分神经元，使得推理结果稳定）；让 Batch Normalization（BN）层使用保存的均值和方差（而不是训练时的动态计算）；计算图 不会保存反向传播的梯度，减少显存占用
    # with torch.no_grad(): ## 关闭 自动梯度计算，减少显存占用，提高推理速度
    #     # voxel的4通道是sdf+RGB
    #     result = model(imgs, img_metas) # 返回tuple,[voxel:torch.Size([1, 4, 60, 100, 20]), voxel feature: torch.Size([1, 64, 60, 100, 20]), torch.Size([1, 304584, 1])]

    # 模拟一个简单的训练循环
    for imgs, img_metas in train_loader:
        # imgs.shape = [B=1, view=3, channels=3, H=518, W=784]
        #  (1,3,3,518,784)
        # 可视化: 将 tensor 转为 numpy 数组，并转换通道顺序以适合 cv2 保存
        # 这里只处理 batch 中第一个样本的所有视角
        # visualimgs = []
        # for view_idx in range(imgs.shape[1]):  # 对每个视角依次处理
        #     # 获取对应视角的图像，形状为 [3, H, W]
        #     image_tensor = imgs[0, view_idx]  
        #     # 转换为 numpy 格式，并将通道从 [3, H, W] 调整为 [H, W, 3]
        #     image_array = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
        #     # 如果图像数据为浮点型且归一化到 [0, 1]，需要乘以255转换为整数
        #     if image_array.dtype != 'uint8':
        #         image_array = (image_array * 255).astype('uint8')
        #     # 转换通道顺序：RGB -> BGR (cv2 保存时使用 BGR)
        #     image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        #     visualimgs.append(image_array)

        # # 将每个视角图像保存到 visual_dir 目录下
        # for i in range(len(visualimgs)):
        #     cv2.imwrite(f'{visual_dir}/{i}.png', visualimgs[i])
        
        imgs = imgs.to(device)
        print("imgs shape:", imgs.shape)
        print("len(img_metas):", len(img_metas))
        print("img_metas keys:", img_metas[0].keys()) # # dict_keys(['pc_range', 'occ_size', 'lidar2img', 'lidar2cam', 'cam_intrinsic', 'img_shape'])
        # 推理模型
        with torch.no_grad():
            result = model(imgs, img_metas) # torch.Size([1, 4, 60, 100, 20])
        # print("result:", result)
        if type(result) == tuple: 
            sdf = result[0][0, 0].detach().squeeze().cpu().numpy() # 最终，显存主要被 model 和 result 占用，推理结束后 result 存在 GPU，除非 .cpu() 取回，否则无法进行可视化
        costmap = sdf2occ(sdf) # sdf2occ() 将 sdf 转换为 occupancy map
        costmap = cv2.rotate(costmap.max(-1), cv2.ROTATE_180) #(60, 100)
        cv2.imwrite(f'{visual_dir}/map.png', costmap * 255)

        # import pdb; pdb.set_trace()
        # save depth
        # 从(1,304584,1) 变成 (3, 259, 392)
        depth = result[-1].view(len(views), input_shape[0] // 2, input_shape[1] // 2).detach().squeeze().cpu().numpy()
        for i in range(len(views)):
            plt.imsave(f'{visual_dir}/depth_{i}.png', np.log10(depth[i]), cmap='jet')
        print('done')
        # import pdb; pdb.set_trace()
        
        break  # 只看一次即可