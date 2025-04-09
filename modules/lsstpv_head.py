import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from modules.occformer.occnet import OccupancyEncoder
from modules.image2bev.ViewTransformerLSSVoxel import (
    ViewTransformerLiftSplatShootVoxel,
)
from modules.render import RaySampler
from modules.render.render_utils import nerf_models
from modules.render.render_utils.rays import RayBundle
from vqvae.vae_2d_resnet import VAERes2D, VAERes3D, VAERes2DImg, VAERes3DVoxel
# from model import LSSTPVDAv2
from vismask.gen_vis_mask import sdf2occ
import matplotlib.pyplot as plt

# def constant_init(module, val, bias=0):
#     if hasattr(module, "weight") and module.weight is not None:
#         nn.init.constant_(module.weight, val)
#     if hasattr(module, "bias") and module.bias is not None:
#         nn.init.constant_(module.bias, bias)


# # 假设已导入 ViewTransformerLiftSplatShootVoxel, OccupancyEncoder, RaySampler, nerf_models, RayBundle
# # 这些模块负责视角变换、占据编码、射线采样和神经渲染等功能
# class LSSTPVHead(nn.Module):
#     def __init__(
#         self,
#         num_classes=17,                 # 类别数量，通常用于预测分类结果（例如语义分割或检测）
#         volume_h=200,                   # 体素体积在高度方向的尺寸（可能可以是列表）
#         volume_w=200,                   # 体素体积在宽度方向的尺寸
#         volume_z=16,                    # 体素体积在深度方向的尺寸
#         upsample_strides=[1, 2, 1, 2],    # 上采样时各层使用的步幅
#         out_indices=[0, 2, 4, 6],         # 在解码器各层中，哪些层的输出用于后续的解码（如跳跃连接）
#         conv_input=None,                # 卷积层的输入通道数（一般为列表，对应每个解码层）
#         conv_output=None,               # 卷积层的输出通道数（一般为列表，每个解码层后的通道数）
#         pad_size=[736, 1280],           # 输入图像的 pad 尺寸
#         downsample=16,                  # 图像特征下采样倍数
#         numC_input=512,                 # 输入转换器通道数
#         numC_Trans=128,                 # 图像视角转换后得到的通道数
#         render_scale=2,                 # 渲染时的图像缩放倍数
#         ray_sample_mode='fixed',        # 射线采样模式，如固定（fixed）
#         num_imgs=3,                     # 多视角图片的数量
#     ):
#         super().__init__()

#         # 定义点云范围：xmin, ymin, zmin, xmax, ymax, zmax
#         point_cloud_range = [0., -10., 0., 12., 10., 4.]
#         voxel_size = 0.2               # 每个体素的尺寸（单位：米）
#         lss_downsample = [1, 1, 1]       # LSS方法的下采样因子，这里各轴均为 1
#         data_config = {"pad_size": pad_size}  # 数据配置，主要记录图像 pad 尺寸

#         # 设置用于 LSS 投射（Lift-Splat-Shoot）方法的 3D 空间网格配置
#         grid_config = {
#             "xbound": [
#                 point_cloud_range[0],
#                 point_cloud_range[3],
#                 voxel_size * lss_downsample[0],
#             ],
#             "ybound": [
#                 point_cloud_range[1],
#                 point_cloud_range[4],
#                 voxel_size * lss_downsample[1],
#             ],
#             "zbound": [
#                 point_cloud_range[2],
#                 point_cloud_range[5],
#                 voxel_size * lss_downsample[2],
#             ],
#             # dbound 可能用于深度或距离分层
#             "dbound": [0.05, 10, 0.05],
#         }

#         # 初始化图像视角转换模块，将图像特征提升到 3D 体素网格上
#         self.img_view_transformer = ViewTransformerLiftSplatShootVoxel(
#             grid_config=grid_config,
#             data_config=data_config,
#             numC_Trans=numC_Trans,
#             vp_megvii=False,
#             downsample=downsample,
#             numC_input=numC_input,
#         )

#         # 分别定义三个占据编码器，分别对 xy、yz、xz 三个视角求解 BEV 特征
#         self.img_bev_encoder_backbone_xy = OccupancyEncoder(
#             num_stage=4,
#             in_channels=numC_Trans,
#             block_numbers=[2, 2, 2, 2],
#             block_inplanes=[numC_Trans, 128, 256, 512],
#             block_strides=[1, 2, 2, 2],
#             out_indices=(0, 1, 2, 3),
#         )

#         self.img_bev_encoder_backbone_yz = OccupancyEncoder(
#             num_stage=4,
#             in_channels=numC_Trans,
#             block_numbers=[2, 2, 2, 2],
#             block_inplanes=[numC_Trans, 128, 256, 512],
#             block_strides=[1, 2, 2, 2],
#             out_indices=(0, 1, 2, 3),
#         )

#         self.img_bev_encoder_backbone_xz = OccupancyEncoder(
#             num_stage=4,
#             in_channels=numC_Trans,
#             block_numbers=[2, 2, 2, 2],
#             block_inplanes=[numC_Trans, 128, 256, 512],
#             block_strides=[1, 2, 2, 2],
#             out_indices=(0, 1, 2, 3),
#         )

#         # 保存构造函数传入的一些参数
#         self.numC_Trans = numC_Trans
#         self.conv_input = conv_input
#         self.conv_output = conv_output

#         self.num_classes = num_classes
#         self.volume_h = volume_h
#         self.volume_w = volume_w
#         self.volume_z = volume_z

#         self.upsample_strides = upsample_strides
#         self.out_indices = out_indices

#         # 定义射线采样器，用于后续渲染过程
#         self.ray_sampler = RaySampler(
#             ray_sample_mode=ray_sample_mode,
#             ray_number=[pad_size[0] // render_scale, pad_size[1] // render_scale],
#             ray_img_size=pad_size,
#         )
        
#         # 定义渲染模块（例如 NeuS 模型）
#         self.render_model = getattr(nerf_models, "NeuSModel")(
#             pc_range=np.array(point_cloud_range, dtype=np.float32),
#             voxel_size=np.array([voxel_size, voxel_size, voxel_size], dtype=np.float32),
#             voxel_shape=np.array([volume_h[0], volume_w[0], volume_z[0]], dtype=np.float32),
#             norm_scene=True,
#             field_cfg=dict(
#                 type="SDFField",
#                 beta_init=0.3,
#             ),
#             collider_cfg=dict(
#                 type="AABBBoxCollider", 
#                 near_plane=0.5,
#             ),
#             sampler_cfg=dict(
#                 type="NeuSSampler",
#                 initial_sampler="UniformSampler",
#                 num_samples=72,
#                 num_samples_importance=24,
#                 num_upsample_steps=1,
#                 train_stratified=True,
#                 single_jitter=True,
#             ),
#             loss_cfg=None,
#         )
        
#         self.num_imgs = num_imgs

#         # 初始化上采样、SDF 和特征融合的网络层
#         self._init_layers()
#         self.vqvae = VAERes2DImg(inp_channels=80, out_channels=80, z_channels=4, mid_channels=320) #!!!
#         # todo

#     # ----------------------------------------------
#     # _init_layers：定义上采样层、SDF 解码层和融合权重获取层
#     def _init_layers(self):
#         self.deblocks = nn.ModuleList()  # 用于存放上采样/解码模块
#         upsample_strides = self.upsample_strides

#         out_channels = self.conv_output
#         in_channels = self.conv_input

#         # 根据通道数和上采样步幅构造每个解码层
#         for i, out_channel in enumerate(out_channels):
#             stride = upsample_strides[i]
#             if stride > 1:
#                 # 若步幅大于1，使用ConvTranspose3d实现上采样操作
#                 upsample_layer = nn.ConvTranspose3d(
#                     in_channels=in_channels[i],
#                     out_channels=out_channel,
#                     kernel_size=upsample_strides[i],
#                     stride=upsample_strides[i],
#                     bias=False,
#                 )
#             else:
#                 # 否则使用常规3d卷积
#                 upsample_layer = nn.Conv3d(
#                     in_channels=in_channels[i],
#                     out_channels=out_channel,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                     bias=False,
#                 )

#             # 使用BatchNorm与ReLU激活对上采样层进行封装，形成解码器块
#             deblock = nn.Sequential(
#                 upsample_layer, nn.BatchNorm3d(out_channel), nn.ReLU(inplace=True)
#             )
#             self.deblocks.append(deblock)

#         # 构造 SDF 解码层：对输出特征进行 1x1x1 卷积，输出类别数的预测
#         self.sdf = nn.ModuleList()
#         for i in self.out_indices:
#             sdf = nn.Conv3d(
#                 in_channels=out_channels[i],
#                 out_channels=self.num_classes,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#                 bias=False,
#             )
#             self.sdf.append(sdf)

#         # 定义三个融合权重获取层，用于分别对 xy、yz、xz 三个视角进行加权融合
#         self.combine_xy = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)
#         self.combine_yz = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)
#         self.combine_xz = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)

#     # 这个 _init_layers 的目的是初始化网络中用于上采样（deblocks）、SDF 预测和多视角特征融合的各个层。

#     # ----------------------------------------------
#     # init_weights：初始化网络中部分特殊层的权重
#     def init_weights(self):
#         """Initialize weights of the DeformDETR head."""
#         for m in self.modules():
#             # 如果模块中存在 DeformConv2dPack 或 ModulatedDeformConv2dPack，则把卷积偏置初始化为0
#             if hasattr(m, "conv_offset"):
#                 constant_init(m.conv_offset, 0)

#     # ----------------------------------------------
#     # forward 函数
#     # 输入:
#     #    image_feats: 来自多视角图像的特征
#     #    poses: 相机外参（如位姿），用于从图像坐标转换到3D空间
#     #    img_metas: 每帧图像的元数据字典（包含 lidar2img，lidar2cam 等信息）
#     #    return_features: 是否返回中间特征（默认False）
#     # 输出:
#     #    sdf_preds[-1]: 最后的3D SDF预测
#     #    outputs[-1]: 最后的解码器输出特征
#     #    render_depths[-1]: 渲染得到的深度图
#     def forward(self, image_feats, poses, img_metas, return_features=False):
#         # ===== 1. 利用视角转换器获取 TPV（三视图体素）特征 =====
#         # 根据相机位姿获取 MLP 输入
#         mlp_input = self.img_view_transformer.get_mlp_input(*poses)  # 此处 *poses 将姿态参数传入
#         geo_inputs = [*poses, mlp_input]  # 组合姿态及 MLP 输入等几何信息
        
#         # 利用视角转换器将图像特征转换到 3D 空间，返回的 xyz_feats 是三维特征网格，depth 为深度信息
#         xyz_feats, depth = self.img_view_transformer([image_feats] + geo_inputs)
#         # 本模块的目的是：将来自多视角的图像特征转换成统一的 3D 空间特征表示

#         # ===== 2. 将3D特征在不同平面上池化，构造鸟瞰视图特征 =====
#         # xy 平面特征：对深度维度（dim=4）求平均
#         xy_feats = xyz_feats.mean(dim=4, keepdim=True)
#         # yz 平面特征：对 x 方位维度（dim=2）求平均
#         yz_feats = xyz_feats.mean(dim=2, keepdim=True)
#         # xz 平面特征：对 y 方位维度（dim=3）求平均
#         xz_feats = xyz_feats.mean(dim=3, keepdim=True)
#         # 目的：挤压3D体素特征，在不同平面（xy, yz, xz）上获得单通道或少量通道的 BEV 特征

#         # ===== 3. 分别优化不同平面的特征 =====
#         # 通过各自的 BEV 编码器提取更高级的特征
#         xy_feats = self.img_bev_encoder_backbone_xy(xy_feats)
#         # 对于 yz 和 xz 特征，需要先对维度做变换以适应编码器的输入格式
#         yz_feats = self.img_bev_encoder_backbone_yz(yz_feats.permute(0, 1, 3, 4, 2))
#         xz_feats = self.img_bev_encoder_backbone_xz(xz_feats.permute(0, 1, 2, 4, 3))
#         # 目的：对三个方向的特征分别进行深层网络的优化提取更有区分性的特征

#         # ===== 4. 融合三个平面的特征 =====
#         volume_embed_reshape = []  # 用于保存每个尺度下融合后的体素特征
#         for i in range(len(xy_feats)):
#             # 扩展 xy 特征使得其在深度维度上达到目标尺寸
#             tpv_xy = (
#                 xy_feats[i]
#                 .permute(0, 1, 2, 3, 4)
#                 .expand(-1, -1, -1, -1, self.volume_z[i])
#             )
#             # 扩展 yz 特征使得其在高度上达到目标尺寸
#             tpv_yz = (
#                 yz_feats[i]
#                 .permute(0, 1, 4, 2, 3)
#                 .expand(-1, -1, self.volume_h[i], -1, -1)
#             )
#             # 扩展 xz 特征使得其在宽度上达到目标尺寸
#             tpv_xz = (
#                 xz_feats[i]
#                 .permute(0, 1, 2, 4, 3)
#                 .expand(-1, -1, -1, self.volume_w[i], -1)
#             )
            
#             # 计算每个视角的融合权重（注意 softmax 在不同维度上计算）
#             xy_coeff = F.softmax(self.combine_xy(xyz_feats), dim=4)
#             yz_coeff = F.softmax(self.combine_yz(xyz_feats), dim=2)
#             xz_coeff = F.softmax(self.combine_xz(xyz_feats), dim=3)
            
#             # 对第一个尺度附加原始特征做残差连接，其余尺度仅进行加权融合
#             if i == 0:
#                 fused = (
#                     tpv_xy * xy_coeff
#                     + tpv_yz * yz_coeff
#                     + tpv_xz * xz_coeff
#                     + xyz_feats
#                 )
#             else:
#                 fused = tpv_xy * xy_coeff + tpv_yz * yz_coeff + tpv_xz * xz_coeff

#             # 将原始 3D 特征进行下采样，为下一级特征对齐做准备（注意 scale_factor 0.5 对尺寸进行缩减）
#             scale_factor = 0.5
#             output_size = [math.ceil(dim * scale_factor) for dim in xyz_feats.shape[2:]]
#             xyz_feats = F.interpolate(xyz_feats, size=output_size, mode="trilinear")
            
#             # 将当前尺度融合后的特征保存，后续用于 skip 连接
#             volume_embed_reshape.append(fused)
#         # 目的：将来自三个视角的信息进行空间上融合，得到更具区分性的3D体素嵌入

#         # ===== 5. 利用上采样/解码层构建最终的3D特征 =====
#         # !obtain 3d features
#         outputs = []  # 存储解码器中部分尺度的输出
#         result = volume_embed_reshape.pop()  # 选取最高尺度（最后一层）的融合特征
#         for i in range(len(self.deblocks)):
#             # 对当前融合特征进行解码（上采样）
#             result = self.deblocks[i](result)
#             if i in self.out_indices:
#                 # 如果当前层索引在需要输出的索引中，则保存该尺度特征
#                 outputs.append(result)
#             elif i < len(self.deblocks) - 1:
#                 # 如果不是最底层，则进行跳跃连接：将当前结果与之前保存的对应尺度的特征进行相加
#                 volume_embed_temp = volume_embed_reshape.pop()
#                 if result.shape != volume_embed_temp.shape:
#                     result = F.interpolate(
#                         result, size=volume_embed_temp.shape[2:], mode="trilinear"
#                     )
#                 result = result + volume_embed_temp #! torch.Size([1, 64, 60, 100, 20])
#         # 目的：通过解码器层逐步上采样，同时融合 skip 连接信息，得到多尺度的3D特征表示

#         # ===== 6. 解码占据/距离预测 =====
#         # !decode occupancy from 3d features
#         sdf_preds = []  # 用于存放每个尺度的 SDF (Signed Distance Function) 预测
#         for i in range(len(outputs)):
#             # 将每个尺度的特征插值到和最后一层相同的尺寸后，利用 1x1x1 卷积预测类别/距离
#             sdf_pred = self.sdf[i](F.interpolate(outputs[i], size=outputs[-1].shape[2:], mode='trilinear'))
#             sdf_preds.append(sdf_pred.contiguous())
#         # 目的：利用解码器输出的3D特征生成最终的体素预测，如物体的占据状态或者距离信息

#         # import pdb; pdb.set_trace()
#         # ===== 7. 整理相机和点云转换矩阵 =====
#         sdf_preds = sdf_preds[-1:]  # 只保留最后尺度的预测 #!4个尺度总共
        

#         # todo 用vqgan对sdf 进行压缩（编码），压缩后不要进行任何操作，马上解压缩（解码），恢复原来的形状
        
#         # --------------------------------------------------------------------
#         #
#         # 说明：
#         # 1. sdf_preds[-1] 的形状为 (B, 4, 60, 100, 20)，重排后通道数为 4*20 = 80，
#         #    与 VAERes2DImg 中设置的 inp_channels 和 out_channels 保持一致（均为80）。
#         # 2. 经过 vqvae 处理后，重构输出保存在字典的 'logits' 字段中，其形状与原输入一致。

#         # vqvae_out = self.vqvae(sdf_preds[-1])  # 调用 VAE 压缩模块
#         # compressed_feature = vqvae_out['mid']     # 取出中间的压缩结果
#         # reconstructed_sdf = vqvae_out['logits']  # 获取解码后的重构结果，其形状为 (B, 4, 60, 100, 20)
#         # # 使用重构后的 sdf 替换原来的 sdf_preds[-1]，便于后续渲染流程观察重建效果
#         # sdf_preds[-1] = reconstructed_sdf

        
#         # 返回最终的 SDF 预测、解码器最后的输出特征以及渲染得到的深度图
#         # import pdb; pdb.set_trace()
#         return sdf_preds[-1]
#         # forward 函数总体目的：从多视角图像输入中提取特征，构造 3D 体素表示，
#         # 融合不同平面的特征，并解码出占据（或距离）预测，同时基于射线采样与神经渲染获得深度图，
#         # 最终为后续任务（如三维重建或检测）提供预测结果。

#     # ----------------------------------------------
#     # depth2plane：将预测的射线深度值转换到图像平面深度
#     def depth2plane(self, ray_o, ray_d, depth, lidar2img):
#         # 根据预测的深度和射线方向计算点在激光雷达坐标系中的位置
#         lidar_points = ray_d * depth / self.render_model.scale_factor
#         lidar_points = lidar_points + ray_o / self.render_model.scale_factor
#         # 重塑成 (num_imgs, -1, 3) 的形状
#         lidar_points = lidar_points.view(self.num_imgs, -1, 3)
#         # 在每个点后拼接1，形成齐次坐标，便于进行矩阵变换
#         lidar_points = torch.cat([lidar_points, torch.ones(lidar_points[..., :1].shape).to(lidar_points)], -1)
#         # 将点从激光雷达坐标系转换到图像坐标系
#         img_points = torch.matmul(lidar2img.unsqueeze(1), lidar_points.unsqueeze(-1))
#         # 最后取出深度部分，并做一个下限裁剪
#         return torch.clamp(img_points.squeeze().view(-1, 4)[:, 2:3], min=1e-5)
#         # 目的：将基于射线计算得到的深度值转换为图像平面的深度，方便后续深度图的评估或监督

#     # ----------------------------------------------
#     # sample_rays：依据图像特征和转换矩阵采样射线
#     def sample_rays(self, feats, lidar2cam, lidar2img):
#         batch_ret = []  # 用于保存每个批次的采样结果
#         for bs_idx in range(len(feats)):
#             # 对于当前批次，提取对应的 lidar2img 和计算其逆矩阵以便从图像坐标转换到激光雷达坐标系
#             i_lidar2img = lidar2img[bs_idx]
#             i_img2lidar = torch.inverse(i_lidar2img)
#             # 同时获得 lidar2cam 的逆变换
#             i_cam2lidar = torch.inverse(lidar2cam[bs_idx])
            
#             # 初始化储存当前批次中各个视角的射线起点、方向、深度及图像坐标
#             i_sampled_ray_o, i_sampled_ray_d, i_sampled_depth, i_sampled_coords = (
#                 [],
#                 [],
#                 [],
#                 [],
#             )

#             # 对当前批次中的每个视角进行采样
#             for c_idx in range(self.num_imgs):

#                 j_sampled_all_pts, j_sampled_all_pts_cam = (
#                     [],
#                     [],
#                 )

#                 # -------------------- 采样点过程 --------------------
#                 # 调用射线采样器采样图像平面上的点（这里利用 round 近似取整）
#                 j_sampled_pts_cam = torch.round(self.ray_sampler())
#                 # 将采样点扩展成齐次坐标，一般为 (x, y, depth, 1, 1)
#                 j_sampled_pts_cam = torch.cat(
#                     [j_sampled_pts_cam, torch.ones_like(j_sampled_pts_cam[..., :1]), torch.ones_like(j_sampled_pts_cam[..., :1])],
#                     -1,
#                 )

#                 # 将采样点从图像坐标转换到激光雷达坐标
#                 j_sampled_pts = torch.matmul(
#                     i_img2lidar[c_idx : c_idx + 1],
#                     torch.cat(
#                         [
#                             j_sampled_pts_cam[..., :2] * j_sampled_pts_cam[..., 2:3],
#                             j_sampled_pts_cam[..., 2:],
#                         ],
#                         dim=-1,
#                     ).unsqueeze(-1),
#                 ).squeeze(-1)[..., :3]
#                 j_sampled_all_pts.append(j_sampled_pts)
#                 j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :2])

#                 if len(j_sampled_all_pts) > 0:
#                     # 将当前视角采样的各点合并
#                     j_sampled_all_pts = torch.cat(j_sampled_all_pts, dim=0)
#                     j_sampled_all_pts_cam = torch.cat(j_sampled_all_pts_cam, dim=0)

#                     # 计算激光雷达下的射线起点，该起点为相机到激光雷达转换矩阵获得的平移部分
#                     unscaled_ray_o = i_cam2lidar[c_idx : c_idx + 1, :3, 3].repeat(
#                         j_sampled_all_pts.shape[0], 1
#                     )
#                     # 存储当前视角对应的射线起点和方向（归一化后的向量）
#                     i_sampled_ray_o.append(
#                         unscaled_ray_o * self.render_model.scale_factor
#                     )
#                     i_sampled_ray_d.append(
#                         F.normalize(j_sampled_all_pts - unscaled_ray_o, dim=-1)
#                     )
#                     # 存储图像坐标系下的采样点（xp, yp）
#                     i_sampled_coords.append(
#                         j_sampled_all_pts_cam
#                     )
#                     # 计算射线长度（距离），作为采样深度，并乘上尺度因子
#                     sampled_depth = (
#                         torch.norm(
#                             j_sampled_all_pts - unscaled_ray_o, dim=-1, keepdim=True
#                         )
#                         * self.render_model.scale_factor
#                     )
#                     i_sampled_depth.append(sampled_depth)
            
#             # 将当前批次所有视角的采样结果按照 key 进行合并
#             batch_ret.append(
#                 {
#                     "ray_o": torch.cat(i_sampled_ray_o, dim=0),
#                     "ray_d": torch.cat(i_sampled_ray_d, dim=0),
#                     "depth": torch.cat(i_sampled_depth, dim=0),
#                     "coords": torch.cat(i_sampled_coords, dim=0),
#                 }
#             )

#         return batch_ret
#         # 目的：对每个批次及每个视角采样射线，用于后续渲染模块生成深度图










import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from modules.occformer.occnet import OccupancyEncoder
from modules.image2bev.ViewTransformerLSSVoxel import (
    ViewTransformerLiftSplatShootVoxel,
)
from modules.render import RaySampler
from modules.render.render_utils import nerf_models
from modules.render.render_utils.rays import RayBundle
from vqvae.vae_2d_resnet import VAERes2DImg, VAERes2DImgDirectBC


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class LSSTPVHead(nn.Module):
    def __init__(
        self,
        num_classes=17,
        volume_h=200,
        volume_w=200,
        volume_z=16,
        upsample_strides=[1, 2, 1, 2],
        out_indices=[0, 2, 4, 6],
        conv_input=None,
        conv_output=None,
        pad_size=[736, 1280],
        downsample=16,
        numC_input=512,
        numC_Trans=128,
        render_scale=2,
        ray_sample_mode='fixed',
        num_imgs=3,
    ):
        super(LSSTPVHead, self).__init__()

        point_cloud_range = [0., -10., 0., 12., 10., 4.]
        voxel_size = 0.2
        lss_downsample = [1, 1, 1]
        data_config = {"pad_size": pad_size}
        grid_config = {
            "xbound": [
                point_cloud_range[0],
                point_cloud_range[3],
                voxel_size * lss_downsample[0],
            ],
            "ybound": [
                point_cloud_range[1],
                point_cloud_range[4],
                voxel_size * lss_downsample[1],
            ],
            "zbound": [
                point_cloud_range[2],
                point_cloud_range[5],
                voxel_size * lss_downsample[2],
            ],
            "dbound": [0.05, 10, 0.05],
        }

        self.img_view_transformer = ViewTransformerLiftSplatShootVoxel(
            grid_config=grid_config,
            data_config=data_config,
            numC_Trans=numC_Trans,
            vp_megvii=False,
            downsample=downsample,
            numC_input=numC_input,
        )

        self.img_bev_encoder_backbone_xy = OccupancyEncoder(
            num_stage=4,
            in_channels=numC_Trans,
            block_numbers=[2, 2, 2, 2],
            block_inplanes=[numC_Trans, 128, 256, 512],
            block_strides=[1, 2, 2, 2],
            out_indices=(0, 1, 2, 3),
        )

        self.img_bev_encoder_backbone_yz = OccupancyEncoder(
            num_stage=4,
            in_channels=numC_Trans,
            block_numbers=[2, 2, 2, 2],
            block_inplanes=[numC_Trans, 128, 256, 512],
            block_strides=[1, 2, 2, 2],
            out_indices=(0, 1, 2, 3),
        )

        self.img_bev_encoder_backbone_xz = OccupancyEncoder(
            num_stage=4,
            in_channels=numC_Trans,
            block_numbers=[2, 2, 2, 2],
            block_inplanes=[numC_Trans, 128, 256, 512],
            block_strides=[1, 2, 2, 2],
            out_indices=(0, 1, 2, 3),
        )

        self.numC_Trans = numC_Trans
        self.conv_input = conv_input
        self.conv_output = conv_output

        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z

        self.upsample_strides = upsample_strides
        self.out_indices = out_indices

        # 定义sampler
        self.ray_sampler = RaySampler(
                ray_sample_mode=ray_sample_mode,
                ray_number=[pad_size[0] // render_scale, pad_size[1] // render_scale],
                ray_img_size=pad_size,
                )
        
        # 定义render model
        self.render_model = getattr(nerf_models, "NeuSModel")(
            pc_range=np.array(point_cloud_range, dtype=np.float32),
            voxel_size=np.array([voxel_size, voxel_size, voxel_size], dtype=np.float32),
            voxel_shape=np.array([volume_h[0], volume_w[0], volume_z[0]], dtype=np.float32),
            norm_scene=True,
            field_cfg=dict(
                type="SDFField",
                beta_init=0.3,
                ),
            collider_cfg=dict(
                type="AABBBoxCollider", 
                near_plane=0.5,
                ),
            sampler_cfg=dict(
                type="NeuSSampler",
                initial_sampler="UniformSampler",
                num_samples=72,
                num_samples_importance=24,
                num_upsample_steps=1,
                train_stratified=True,
                single_jitter=True,
                ),
            loss_cfg=None,
            )
        
        self.num_imgs = num_imgs
        self._init_layers()
        self.vqvae = VAERes2DImgDirectBC(inp_channels=80, out_channels=80, z_channels=4, mid_channels=1024) #!!!


    def _init_layers(
        self,
    ):
        self.deblocks = nn.ModuleList()
        upsample_strides = self.upsample_strides

        out_channels = self.conv_output
        in_channels = self.conv_input

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = nn.ConvTranspose3d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False,
                )
            else:
                upsample_layer = nn.Conv3d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )

            deblock = nn.Sequential(
                upsample_layer, nn.BatchNorm3d(out_channel), nn.ReLU(inplace=True)
            )

            self.deblocks.append(deblock)

        self.sdf = nn.ModuleList()
        for i in self.out_indices:
            sdf = nn.Conv3d(
                in_channels=out_channels[i],
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.sdf.append(sdf)

        self.combine_xy = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)
        self.combine_yz = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)
        self.combine_xz = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, "conv_offset"):
                constant_init(m.conv_offset, 0)

    def forward(self, image_feats, poses, img_metas, return_features=False):
        # get tpv features based on lss
        mlp_input = self.img_view_transformer.get_mlp_input(*poses)
        geo_inputs = [*poses, mlp_input]
        xyz_feats, depth = self.img_view_transformer([image_feats] + geo_inputs)

        xy_feats = xyz_feats.mean(dim=4, keepdim=True)
        yz_feats = xyz_feats.mean(dim=2, keepdim=True)
        xz_feats = xyz_feats.mean(dim=3, keepdim=True)

        # optimize tpv features separately
        xy_feats = self.img_bev_encoder_backbone_xy(xy_feats)
        yz_feats = self.img_bev_encoder_backbone_yz(yz_feats.permute(0, 1, 3, 4, 2))
        xz_feats = self.img_bev_encoder_backbone_xz(xz_feats.permute(0, 1, 2, 4, 3))

        # fuse tpv features
        volume_embed_reshape = []
        for i in range(len(xy_feats)):
            tpv_xy = (
                xy_feats[i]
                .permute(0, 1, 2, 3, 4)
                .expand(-1, -1, -1, -1, self.volume_z[i])
            )
            tpv_yz = (
                yz_feats[i]
                .permute(0, 1, 4, 2, 3)
                .expand(-1, -1, self.volume_h[i], -1, -1)
            )
            tpv_xz = (
                xz_feats[i]
                .permute(0, 1, 2, 4, 3)
                .expand(-1, -1, -1, self.volume_w[i], -1)
            )
            xy_coeff = F.softmax(self.combine_xy(xyz_feats), dim=4)
            yz_coeff = F.softmax(self.combine_yz(xyz_feats), dim=2)
            xz_coeff = F.softmax(self.combine_xz(xyz_feats), dim=3)
            if i == 0:
                fused = (
                    tpv_xy * xy_coeff
                    + tpv_yz * yz_coeff
                    + tpv_xz * xz_coeff
                    + xyz_feats
                )
            else:
                fused = tpv_xy * xy_coeff + tpv_yz * yz_coeff + tpv_xz * xz_coeff

            scale_factor = 0.5
            output_size = [math.ceil(dim * scale_factor) for dim in xyz_feats.shape[2:]]
            xyz_feats = F.interpolate(xyz_feats, size=output_size, mode="trilinear")
            volume_embed_reshape.append(fused)

        # obtain 3d features
        outputs = []
        result = volume_embed_reshape.pop()
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)
            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks) - 1:  # we do not add skip connection at level 0
                volume_embed_temp = volume_embed_reshape.pop()
                if result.shape != volume_embed_temp.shape:
                    result = F.interpolate(
                        result, size=volume_embed_temp.shape[2:], mode="trilinear"
                    )
                result = result + volume_embed_temp
        
        # decode occupancy from 3d features
        sdf_preds = []
        for i in range(len(outputs)):
            sdf_pred = self.sdf[i](F.interpolate(outputs[i], size=outputs[-1].shape[2:], mode='trilinear'))
            sdf_preds.append(sdf_pred.contiguous())

        # obtain lidar2cam and lidar2img
        sdf_preds = sdf_preds[-1:]

        # todo 用vqgan对sdf 进行压缩（编码），压缩后不要进行任何操作，马上解压缩（解码），恢复原来的形状
        
        # todo --------------------------------------------------------------------
        #
        # 说明：
        # 1. sdf_preds[-1] 的形状为 (B, 4, 60, 100, 20)，重排后通道数为 4*20 = 80，
        #    与 VAERes2DImg 中设置的 inp_channels 和 out_channels 保持一致（均为80）。
        # 2. 经过 vqvae 处理后，重构输出保存在字典的 'logits' 字段中，其形状与原输入一致。

        vqvae_out = self.vqvae(sdf_preds[-1])  # 调用 VAE 压缩模块
        compressed_feature = vqvae_out['mid']     # 取出中间的压缩结果
        reconstructed_sdf = vqvae_out['logits']  # 获取解码后的重构结果，其形状为 (B, 4, 60, 100, 20)
        # 使用重构后的 sdf 替换原来的 sdf_preds[-1]，便于后续渲染流程观察重建效果
        recon_loss = F.mse_loss(reconstructed_sdf, sdf_preds[0])  # 与输入 voxel 做 MSE
        print(recon_loss)
        sdf_preds[-1] = reconstructed_sdf

    

        lidar2img, lidar2cam = [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
        lidar2img = torch.from_numpy(np.asarray(lidar2img)).to(image_feats)
        lidar2cam = torch.from_numpy(np.asarray(lidar2cam)).to(image_feats)
        
        # render depth
        rays = self.sample_rays(image_feats, lidar2cam, lidar2img)
        render_depths = []
        for i in range(len(sdf_preds)):
            render_depth = []
            for bs_idx in range(len(lidar2cam)):
                i_ray_o, i_ray_d, i_ray_depth, scaled_points = (    # i_ray_o: n * num_points, 3 (6 * 512, 3)
                    rays[bs_idx]["ray_o"],
                    rays[bs_idx]["ray_d"],
                    rays[bs_idx].get("depth", None),
                    rays[bs_idx].get("scaled_points", None)
                )
                ray_bundle = RayBundle(
                        origins=i_ray_o, directions=i_ray_d, depths=None # i_ray_depth
                    )

                preds_dict = self.render_model(
                    ray_bundle, sdf_preds[i][bs_idx].permute(0, 3, 2, 1).contiguous(), points=scaled_points)

                # 点到相机平面的depth
                render_depth.append(self.depth2plane(i_ray_o, i_ray_d, preds_dict['depth'], lidar2img[bs_idx]))

            render_depths.append(torch.stack(render_depth, 0))

        # return *sdf_preds, depth
        return sdf_preds[-1], outputs[-1], render_depths[-1]


    def depth2plane(self, ray_o, ray_d, depth, lidar2img):
        lidar_points = ray_d * depth / self.render_model.scale_factor
        lidar_points = lidar_points + ray_o / self.render_model.scale_factor
        lidar_points = lidar_points.view(self.num_imgs, -1, 3)
        lidar_points = torch.cat([lidar_points, torch.ones(lidar_points[..., :1].shape).to(lidar_points)], -1)
        img_points = torch.matmul(lidar2img.unsqueeze(1), lidar_points.unsqueeze(-1))
        return torch.clamp(img_points.squeeze().view(-1, 4)[:, 2:3], min=1e-5)
    
    
    def sample_rays(self, feats, lidar2cam, lidar2img):
        batch_ret = []
        for bs_idx in range(len(feats)):
            i_lidar2img = lidar2img[bs_idx] # .flatten(0, 1)
            i_img2lidar = torch.inverse(
                i_lidar2img
            )  # TODO: Are img2lidar and img2cam consistent after image data augmentation?
            i_cam2lidar = torch.inverse(
                lidar2cam[bs_idx] # .flatten(0, 1)
            )

            i_sampled_ray_o, i_sampled_ray_d, i_sampled_depth, i_sampled_coords = (
                [],
                [],
                [],
                [],
            )

            for c_idx in range(self.num_imgs):

                j_sampled_all_pts, j_sampled_all_pts_cam = (
                    [],
                    [],
                )

                """ sample points """
                j_sampled_pts_cam = torch.round(self.ray_sampler())
                j_sampled_pts_cam = torch.cat([j_sampled_pts_cam, torch.ones_like(j_sampled_pts_cam[..., :1]), torch.ones_like(j_sampled_pts_cam[..., :1])], -1)

                j_sampled_pts = torch.matmul(
                    i_img2lidar[c_idx : c_idx + 1],
                    torch.cat(
                        [
                            j_sampled_pts_cam[..., :2]
                            * j_sampled_pts_cam[..., 2:3],
                            j_sampled_pts_cam[..., 2:],
                        ],
                        dim=-1,
                    ).unsqueeze(-1),
                ).squeeze(-1)[..., :3]
                j_sampled_all_pts.append(j_sampled_pts)
                j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :2])

                if len(j_sampled_all_pts) > 0:
                    """merge"""
                    j_sampled_all_pts = torch.cat(j_sampled_all_pts, dim=0)
                    j_sampled_all_pts_cam = torch.cat(j_sampled_all_pts_cam, dim=0)

                    unscaled_ray_o = i_cam2lidar[c_idx : c_idx + 1, :3, 3].repeat(
                        j_sampled_all_pts.shape[0], 1
                    )
                    i_sampled_ray_o.append(
                        unscaled_ray_o * self.render_model.scale_factor
                    )
                    i_sampled_ray_d.append(
                        F.normalize(j_sampled_all_pts - unscaled_ray_o, dim=-1)
                    )
                    i_sampled_coords.append(
                        j_sampled_all_pts_cam
                    )
                    sampled_depth = (
                        torch.norm(
                            j_sampled_all_pts - unscaled_ray_o, dim=-1, keepdim=True
                        )
                        * self.render_model.scale_factor
                    )
                    i_sampled_depth.append(sampled_depth)
            
            batch_ret.append(
                {
                "ray_o": torch.cat(i_sampled_ray_o, dim=0),
                "ray_d": torch.cat(i_sampled_ray_d, dim=0),
                "depth": torch.cat(i_sampled_depth, dim=0),
                "coords": torch.cat(i_sampled_coords, dim=0),
                }
                )

        return batch_ret