import json
import cv2
import numpy as np
import pickle
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
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


class MyDatasetOnlyforVAE(Dataset):
    def __init__(self, args):
        
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
        self.input_shape = [args.input_height, args.input_width]
        
        # 使用的相机视角列表，根据之前的推理代码
        self.views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.concat = args.concat
    
    def __len__(self):
        """
        返回数据集总帧数（并非总样本数，因为一条样本可能由4帧组成）。
        """
        return len(self.data)

    def __getitem__(self, idx):
        if ((idx + self.tem * (self.frames - 1)) >= len(self)) or \
           (self.data[idx]['scene_token'] != self.data[idx + self.tem * (self.frames - 1)]['scene_token']):
            idx = idx - self.tem * (self.frames - 1)

        # 只会得到 [idx] 一个索引
        index_list = [idx + i * self.tem for i in range(self.frames)]

        # ==============================
        # 宏观解释：接下来，我们会把这4帧中，每一帧涉及到的3个相机图像全部读进来，并做和推理时
        # 一样的预处理，包括通道变换、归一化、尺寸插值、调整内参等。
        # 我们还会构造 img_metas，把 ego2imgs, ego2cams, cam_intrinsic 等都存进去。
        # ==============================
        
        # 用于存储所有帧的图像张量 (4帧 × 3视角)
        # 注意：最后我们会把它组织成一个 torch.Tensor 交给DataLoader
        all_imgs = []
        
        
        # 遍历这4帧，在单帧情况下，该循环只跑1次，即 f_idx = idx
        for f_idx in index_list:
            item_info = self.data[f_idx]
            
            # 取出当前帧所在的 scene_token, timestamp 等信息（若后续需要可保留）
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
                
                # ==============================
                # 新增内容：读取深度图并拼接到RGB图像上，形成RGB-D图像，通道数由3变为4
                # ==============================
                depth_path = cam_data['depth_path']  # 获取深度图路径
                # 如果路径以 /data 开头，则更换为绝对路径
                if depth_path.startswith("data/"):
                    depth_path = depth_path.replace("data", "/mnt/bn/occupancy3d/real_world_data/preprocess", 1)
                # 使用cv2.IMREAD_UNCHANGED读取深度图，不做BGR转换
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_img is None:
                    # 如果深度图路径异常，则抛出异常
                    raise ValueError(f"Fail to read depth image: {depth_path}")
                # 如果深度图为单通道，则扩展最后的维度，使形状变为(H, W, 1)
                if len(depth_img.shape) == 2:
                    depth_img = np.expand_dims(depth_img, axis=-1)  # 插入一个通道维度
                # 将深度图转换为float32，并归一化到[0,1]（假定原始深度图像素范围为[0,255]）
                depth_img = depth_img.astype(np.float32) / 1000.0 #!
                # 将归一化后的RGB图像和深度图拼接在一起，形成RGB-D图像，通道数从3变为4
                img_concat = np.concatenate([img_normalized, depth_img], axis=-1)
                if self.concat == True:
                    print("channel concat...")
                    img_normalized = img_concat  # 更新img_normalized，使后续处理得到RGB-D图像 
                
                # 注意：现阶段还未做resize (后面我们用torch的F.interpolate统一裁到 self.input_shape)
                frame_imgs.append(img_normalized)  # todo shape: (H, W, 4(RGBD))
                
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
            
            # 把该帧的3张图像拼成一个 np.array，以便后续转 torch
            # shape: (3, H, W, 3) —— 因为3个视角
            frame_imgs_np = np.stack(frame_imgs, axis=0)
            
            # 保存到all_imgs里
            all_imgs.append(frame_imgs_np)
 
     
        # 此时 all_imgs 的 shape 原始为 (1, 3, H, W, 4)，因为只有 1 帧、3 视角，每张图像为 4 通道(RGB+D)
        all_imgs_np = np.stack(all_imgs, axis=0)  # (frames=1, 3, H, W, 4)
        
        # 由于 frames=1，可以先 squeeze 掉帧维度，得到 (3, H, W, 4)
        all_imgs_np = np.squeeze(all_imgs_np, axis=0)  # shape: (3, H, W, 4)
        
        # 接下来，将视角维度（3）作为时间帧，通道数（4）放在第一维，即转换为 (channels, temporal, H, W)
        # 这里的转换将最后一个维度（4）移动到最前面，维持后面的顺序不变
        all_imgs_np = np.transpose(all_imgs_np, (3, 0, 1, 2))  # shape: (4, 3, H, W)
        
        # 转成 torch.Tensor
        all_imgs_torch = torch.from_numpy(all_imgs_np).float()
        
        # 利用 F.interpolate 对空间尺寸进行插值（这里需要将 (channels, temporal, H, W) 看成多个 2D 图像来处理）
        C, T, H, W = all_imgs_torch.shape  # C=4, T=3
        # Resize 后 shape: (4, 3, H, W)

        # 简单重复，扩展 temporal 维到 17（即第二维扩展为17）
        if T != 17:
            repeat_factor = 17 // T
            remainder = 17 % T
            repeated = [all_imgs_torch.repeat(1, repeat_factor, 1, 1)]
            if remainder > 0:
                repeated.append(all_imgs_torch[:, :remainder])
            all_imgs_torch = torch.cat(repeated, dim=1)

        all_imgs_torch = all_imgs_torch.reshape(C * T, 1, H, W)  # => [12, 1, 1380, 1920]
        all_imgs_torch = F.interpolate( #!必须是(N，C，H，W)
            all_imgs_torch, 
            size=(self.input_shape[0], self.input_shape[1]),
            mode='bilinear',
            align_corners=False
        )
        all_imgs_torch = all_imgs_torch.reshape(C, T, self.input_shape[0], self.input_shape[1])  # 恢复 shape: (4, 3, 目标高, 目标宽)
        
        
        # 最终返回的 tensor 形状为 (channels, temporal, 高, 宽)，即 (4, 3, 目标高, 目标宽)
        return all_imgs_torch


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

    def __init__(self, args):
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
        
        # 使用的相机视角列表，根据之前的推理代码
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





class MyDataset(Dataset):
    '''
    从多帧图像和传感器数据中，构造用于感知和预测的输入样本
    也即：
        读取数据（图像+传感器数据）。
        处理图像（归一化、调整尺寸）。
        计算相机参数（外参、内参）。
        计算自车位姿变换（x, y, yaw）。
        返回处理后的数据。
    '''
    def __init__(self):
        self.tem = 5 # 代表时间步长（temporal step），用于索引数据时的时间间隔
        self.frames = 4 # 代表一个样本由 4 个连续帧组成
        # self.data = pickle.load(open('/mnt/bn/occupancy3d/workspace/lzy/Occ3d/pkls/dcr_data_dynamic_bottom.pkl', '+rb'))['infos'] # 通过 pickle 加载的数据，来源于 dcr_data_dynamic_bottom.pkl，其中 infos 存储了所有数据帧的信息 
        self.data = pickle.load(open('/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_data_bottom_static_scaleddepth_240925_sub.pkl', '+rb'))['infos']
        self.mean = np.array([0.485, 0.456, 0.406])[None, None, :] # 这是用于标准化图像的均值和标准差，通常用于归一化 RGB 图像，每个数值对应 ImageNet 预训练模型的均值和标准差
        self.std = np.array([0.229, 0.224, 0.225])[None, None, :]
        self.input_shape = [518, 784]
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # 确保索引合法
        if ((idx + self.tem * (self.frames - 1)) >= len(self)) or (self.data[idx]['scene_token'] != self.data[idx + self.tem * (self.frames - 1)]['scene_token']):
            idx -= self.tem * (self.frames - 1)
        
        # get data infos 选取 self.frames 个连续数据帧
        infos = []
        for i in range(self.frames):
            infos.append(self.data[idx + self.tem * i])
        
        # get images, extrinsics, intrinsics
        imgs = [] # 存储当前帧的 RGB 图像，经过 归一化 和 标准化 处理
        lidar2imgs = [] # Lidar 到 Camera 的外参矩阵（cam_extrinsic 的逆矩阵）
        lidar2cams = [] # Lidar 到 Image 的变换矩阵，即 scale_factor @ viewpad @ lidar2cam
        cam_intrinsics = [] # 存储当前帧的 Camera 内参矩阵
        for i, info in enumerate(infos):
            for cam_type, cam_info in info['cams'].items(): # 遍历相机不同视角
                img_path = cam_info['data_path']
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to read image: {img_path}")
                ori_shape = img.shape[:2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
                # 归一化 img / 255.0，将像素值缩放到 [0,1]。
                # 标准化 (img - mean) / std，减去均值再除以标准差。   
                imgs.append((img / 255.0 - self.mean) / self.std)

                scale_factor = np.eye(4) # scale_factor 是图像缩放变换矩阵，用于调整内参矩阵
                scale_factor[0, 0] *= self.input_shape[1] / ori_shape[1]
                scale_factor[1, 1] *= self.input_shape[0] / ori_shape[0]
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4) # viewpad 是 4x4 的扩展内参矩阵（原始内参是 3x3） 
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                cam_intrinsics.append(scale_factor @ viewpad) # scale_factor @ viewpad 计算缩放后的相机内参

                lidar2cam = np.linalg.inv(cam_info['cam_extrinsic'])
                lidar2cams.append(lidar2cam) # cam_extrinsic 是相机到 LiDAR 的外参矩阵，取逆得到 LiDAR → Camera 变换

                lidar2img = (scale_factor @ viewpad @ lidar2cam)
                lidar2imgs.append(lidar2img) # lidar2img 计算 LiDAR → Image 变换，用于点云投影
        
        source = np.asarray(imgs[:len(imgs) // 2]).astype(np.float32)
        target = np.asarray(imgs[len(imgs) // 2:]).astype(np.float32)
        lidar2imgs = np.asarray(lidar2imgs[:len(imgs) // 2]).astype(np.float32)
        lidar2cams = np.asarray(lidar2cams[:len(imgs) // 2]).astype(np.float32)
        cam_intrinsics = np.asarray(cam_intrinsics[:len(imgs) // 2]).astype(np.float32)

        # get x,y,yaw 计算 source → target 的变换矩阵 transform
        source_pose = infos[self.frames // 2 - 1]['ego2global_transformation']
        target_pose = infos[self.frames - 1]['ego2global_transformation']
        transform = np.linalg.inv(source_pose) @ target_pose

        # 从 transform 矩阵提取自车的 x, y, yaw 变化
        yaw = Rotation.from_matrix(transform[:3, :3]).as_euler('zyx', degrees=False).tolist()[0]
        x, y = transform[0, 3], transform[1, 3]
        pose = np.array([x, y, yaw]).astype(np.float32)

        # # get x, y, yaw, for test
        # test_pose = []
        # for i in range(10):
        #     source_pose = self.data[idx + ((i + 1) * self.frames // 2 - 1) * self.tem]['ego2global_transformation']
        #     target_pose = self.data[idx + ((i + 2) * self.frames // 2 - 1) * self.tem]['ego2global_transformation']
        #     transform = np.linalg.inv(source_pose) @ target_pose
        #     yaw = Rotation.from_matrix(transform[:3, :3]).as_euler('zyx', degrees=False).tolist()[0]
        #     x, y = transform[0, 3], transform[1, 3]
        #     tmp_pose = np.array([x, y, yaw]).astype(np.float32)
        #     test_pose.append(tmp_pose)
        # test_pose = np.asarray(test_pose).astype(np.float32)


        # jpg：目标图像（target）。
        # txt：目标位姿信息 [x, y, yaw]（pose）。
        # hint：输入图像（source）。
        # lidar2imgs：LiDAR 到图像的变换矩阵。
        # lidar2cams：LiDAR 到相机的变换矩阵。
        # cam_intrinsics：相机内参。
        return dict(
            jpg=target, 
            txt=pose[None, :], 
            hint=source, 
            lidar2imgs=lidar2imgs, 
            lidar2cams=lidar2cams, 
            cam_intrinsics=cam_intrinsics, 
            # test_pose=test_pose,
        )

