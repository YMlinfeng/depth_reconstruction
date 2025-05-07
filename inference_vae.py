# -*- coding: utf-8 -*-
import os
import cv2
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import AutoencoderKLCogVideoX

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="推理脚本参数设置")
    parser.add_argument("--visual_dir", type=str, default='./output/d425/vae3/',
                        help="保存可视化结果的输出路径")
    parser.add_argument("--jsonl_file", type=str,
                        default='/mnt/bn/pretrain3d/real_word_data/preprocess/jsonls/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec.jsonl',
                        help="包含样本信息的 JSON Lines 文件路径，读取第一行数据")
    parser.add_argument("--vqvae_weight", type=str,
                        default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints_vae_c3/vqvae_epoch1_step2000.pth",
                        help="3D Casual VAE（VQ-VAE）模型权重文件路径")
    parser.add_argument("--input_height", type=int, default=520,
                        help="图像输入高度")
    parser.add_argument("--input_width", type=int, default=784,
                        help="图像输入宽度")
    parser.add_argument("--model", type=str, default="Cog3DVAE", help="choices: VAERes2DImgDirectBC, VQModel, Cog3DVAE")
    parser.add_argument("--mode", type=str, default="eval", help="choices: train, eval")
    parser.add_argument("--general_mode", type=str, default="vae", help="choices: vae, vqgan")
    parser.add_argument('--concat', action='store_true', help='是否拼接RGB与Depth，默认不拼接')
    args = parser.parse_args()

    device = get_device()
    print(cv2.__file__)
    print(os.__file__)
    print(f"args: {args}")

    # 保存路径
    visual_dir = args.visual_dir
    os.makedirs(visual_dir, exist_ok=True)

    sample = open(args.jsonl_file, 'r').readlines()[0]  # readlines()[0] 只读取第一个 JSON 对象（通常对应一帧数据）
    sample = json.loads(sample)  # json.loads() 将字符串转换为 Python 字典（dict）

    # 加载前视三目图像
    views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    imgs = [cv2.imread(sample['images'][view]) for view in views]

    depths = []
    for i, view in enumerate(views):
        # 1) 先拿到对应的RGB图像路径
        rgb_path = sample['images'][view]
        # 2) 替换路径中“images”为“depthanythingv2”
        depth_path = rgb_path.replace("images", "depthanythingv2")
        depth_path = depth_path.replace(".jpg", ".png")
        # 3) 读取深度图
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # shape是(H, W)
        depths.append(depth_img)

    # 可视化重建之前的图像和depth
    for i in range(len(imgs)):
        if imgs[i] is None:
            raise ValueError(f"第{i}张RGB图像读取失败，请检查路径！")
        cv2.imwrite(f'{visual_dir}/{i}_rgb_before.png', imgs[i])  # 保留你原本的写法
        if depths[i] is None:
            raise ValueError(f"第{i}张Depth图像读取失败，请检查路径！")
        # 这里为了可视化，假设depth是单通道，可以做一个简单归一化再保存
        depth_vis = depths[i]
        cv2.imwrite(f'{visual_dir}/{i}_depth_before.png', depth_vis)
        depth_vis = depths[i].astype(np.float32) / 1000.0
        depth_norm = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
        depth_vis_uint8 = (depth_norm * 255).astype(np.uint8)
        cv2.imwrite(f'{visual_dir}/{i}_depth_before_d1000.png', depth_vis_uint8)


    # 数据预处理 参考dataset和主训练代码
    # ------
    # 这里我们模拟Dataset中的预处理逻辑：把RGB转成float32并除255.0 -> 转成RGB(若需要) -> 减均值除标准差 -> 与Depth拼接 -> resize等
    # ------
    processed_list = []
    for i in range(len(imgs)):
        # a) 先检查RGB是否成功读取
        if imgs[i] is None:
            raise ValueError(f"RGB图像{i}为空！")
        rgb_bgr = imgs[i].astype(np.float32) / 255.0  # shape (H, W, 3)，BGR格式归一化到[0,1]
        rgb = rgb_bgr[..., [2, 1, 0]]  # 转成RGB通道顺序, shape仍然 (H, W, 3)

        # b) 读取Depth
        depth_data = depths[i]
        if depth_data is None:
            raise ValueError(f"深度图{i}为空！")
        # 若深度图是单通道 shape (H, W)，则扩展为(H, W, 1)
        if len(depth_data.shape) == 2:
            depth_data = np.expand_dims(depth_data, axis=-1).astype(np.float32)  # shape -> (H, W, 1)
        elif depth_data.shape[-1] > 1:
            depth_data = depth_data[..., :1].astype(np.float32)  # 只取一个通道

        # c) 对Depth也归一化到[0,1]，假设它原本是[0,255]量程
        depth_data /= 255.0

        # d) 拼接RGB+D => (H, W, 4) 如果 args.concat 为True，否则仅使用RGB (H, W, 3)
        if args.concat == True:
            rgbd = np.concatenate([rgb, depth_data], axis=-1)  # shape (H, W, 4) 进行RGB和Depth通道拼接
            means = np.array([0.485, 0.456, 0.406, 0.0], dtype=np.float32).reshape(1, 1, 4)  # 每个通道的均值, shape (1,1,4)
            stds  = np.array([0.229, 0.224, 0.225, 1.0], dtype=np.float32).reshape(1, 1, 4)  # 每个通道的标准差, shape (1,1,4)
        else:
            rgbd = rgb  # 只使用RGB数据, shape (H, W, 3)
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)    # 均值, shape (1,1,3)
            stds  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)    # 标准差, shape (1,1,3)

        # e) 与训练一致，需要先减均值除以标准差(针对前3个通道RGB)，注意如果不拼接，则只有RGB 3通道
        rgbd_norm = (rgbd - means) / stds  # shape 不变，(H, W, 4) 或 (H, W, 3)

        # f) resize到(args.input_height, args.input_width)
        rgbd_resized = cv2.resize(
            rgbd_norm,
            (args.input_width, args.input_height),  # (宽, 高)
            interpolation=cv2.INTER_LINEAR
        )  # shape 变为 (input_height, input_width, 4) 或 (input_height, input_width, 3)

        # g) 将 np.transpose: (H,W,C) => (C,H,W)
        rgbd_chw = np.transpose(rgbd_resized, (2, 0, 1))  # shape 变为 (C, input_height, input_width)，其中 C 为拼接后通道数 (3 或 4)

        processed_list.append(rgbd_chw)

    # h) 将三帧(或三视角)拼到一起
    #    如果 args.concat 为 True，则 processed_list 中每个元素的 shape 为 (4, H, W) => stacked tensor shape (3, 4, H, W)
    #    如果 args.concat 为 False，则每个元素的 shape 为 (3, H, W) => stacked tensor shape (3, 3, H, W)
    processed_np = np.stack(processed_list, axis=0)  # shape (T, C, H, W)，其中 T 即视角数=3

    # i) （原注释）再加一个时间frames维度 => shape (3, 4, 1, H, W)
    #    但现在根据训练需求，需要变成 [B=1, C, T, H, W]
    #    所以我们不按原先方式在 axis=2 上插入；而是先转置一下，让维度对上 (C, T, H, W)，然后在最前面插一个 batch=1
    #
    #    #todo 下面是新的处理，保证和训练代码 (1,C,3,H,W) 对齐
    #    注意：此时 processed_np 是 (T, C, H, W)，
    #         我们要把它变成 (C, T, H, W) => 再变成 (1, C, T, H, W)
    processed_np = processed_np.transpose(1, 0, 2, 3)  # 转置后 shape 变为 (C, T, H, W) ——若 C=4 则 (4,3,H,W)，若 C=3 则 (3,3,H,W)
    processed_np = np.expand_dims(processed_np, axis=0)  # 在最前面增加 batch 维度，最终 shape 为 (1, C, T, H, W)

    # j) 现在最终送入模型的tensor形状根据是否拼接不同：
    #    如果 args.concat 为 True，则 tensor shape 为 (1,4,3,520,784)
    #    否则 tensor shape 为 (1,3,3,520,784)
    input_tensor = torch.from_numpy(processed_np).float().to(device)

    #---【张量形状说明】
    # step h) processed_list 里每个元素是 (C, H, W) ，stack 后 => (T, C, H, W)，其中 T=3；C=4或3
    # step i) 经过 transpose(1,0,2,3) => (C, T, H, W)，然后 expand_dims => (1, C, T, H, W)
    #         这样就跟训练代码中 (B, C, T, H, W) 一致。

    #todo 进行推理和重建
    # 新增：根据参数 concat 动态设置模型的输入与输出通道数
    if args.concat:
        # 如果进行RGB与Depth拼接，则最终应使用4通道输入输出
        model_in_channels = 4   # 4通道: RGB (3) + Depth (1)
        model_out_channels = 4
    else:
        # 如果不进行拼接，则只使用RGB，即3通道输入输出
        model_in_channels = 3   # 3通道: RGB only
        model_out_channels = 3

    # 1) 加载3D Casual VAE模型 + 加载checkpoint
    vqvae = AutoencoderKLCogVideoX(
        in_channels=model_in_channels,    # 根据concat参数设置输入通道数；若不拼接则为3通道，若拼接则为4通道
        out_channels=model_out_channels,  # 重建输出通道数，根据拼接设置：RGB或RGBD
        sample_height=args.input_height,
        sample_width=args.input_width,
        latent_channels=4,
        temporal_compression_ratio=4.0,
    )
    # 2) 加载checkpoint (下面一行很关键)
    checkpoint = torch.load(args.vqvae_weight, map_location='cpu')
    print(vqvae.load_state_dict(checkpoint['vqvae_state_dict'], strict=True))
    vqvae.eval()
    vqvae.to(device)

    # 3) 前向推理
    with torch.no_grad():
        encode_out = vqvae.encode(input_tensor, return_dict=True)  # =>返回 AutoencoderKLOutput
        latents = encode_out.latent_dist.mode()  # shape [B, latentC, T', H', W'] = [B, 4, 1, 65, 98] 或 [B, 3, 1, 65, 98] 根据模型通道
        decode_out = vqvae.decode(latents, return_dict=True)
        recons = decode_out.sample  # shape [B, C, T, H, W]，C对应RGBD或RGB，根据输入而定
        #todo 实际通过debug得到的形状可能不同，请确保输出张量形状与预期一致

    # 可视化重建之后的图像和depth
    if args.concat:
        #--- 原有逻辑：当拼接RGB与Depth时，recons输出张量形状为 (1,4,3,input_height,input_width)
        recons_np = recons.cpu().numpy()  # 将 tensor 转为 numpy, shape (1, 4, 3, H, W)
        
        # 为了和之前“可视化时按视角分图”的习惯一致，这里拆分 batch 维度
        recons_np = recons_np[0]  # 变为 (4, 3, H, W) ——4通道: 前3为RGB，最后1为Depth
        
        # 新增：定义RGB反归一化的均值和标准差，形状为 (3, 1, 1)
        mean_rgb = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std_rgb  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        # 对每个视角进行处理，视角沿第2维 (原注释中的 v=0,1,2)
        for v in range(recons_np.shape[1]):
            # 1) 从 (4, H, W) 中选取第v个视角, recon_i 的 shape 为 (4, H, W)
            recon_i = recons_np[:, v, ...]  # shape (4, H, W)
        
            # 2) 分离RGB与Depth
            recon_rgb = recon_i[:3, ...]   # 前3通道为RGB, shape (3, H, W)
            recon_depth = recon_i[3:4, ...] # 第4通道为Depth, shape (1, H, W)
        
            # 3) 反归一化RGB：
            #    预处理时执行了: img_normalized = (img_rgb - mean)/std
            #    此处反操作为: img_denormalized = img_normalized * std + mean
            recon_rgb_denorm = recon_rgb * std_rgb + mean_rgb  # shape (3, H, W)
            recon_rgb_denorm = np.clip(recon_rgb_denorm, 0.0, 1.0)  # 保证值在 [0,1] 范围内
        
            # 4) 将RGB从 (3, H, W) 转换为 (H, W, 3)
            recon_rgb_hw3 = np.transpose(recon_rgb_denorm, (1, 2, 0))  # shape (H, W, 3)
        
            # 5) 将RGB数据乘以255并转换为 uint8 格式
            recon_rgb_uint8 = (recon_rgb_hw3 * 255).astype(np.uint8)
        
            # 6) 对Depth通道进行反向处理
            #    预处理时对depth一般执行了：depth = depth_img.astype(np.float32)/1000.0，且拼接时未进行均值和标准差操作
            #    若目前模型输出的深度通道为 [-1,0]，可以简单将其加1映射到 [0,1]，再根据需要乘一个尺度（例如1000）还原单位
            recon_depth_denorm = recon_depth + 1.0        # shape (1, H, W)  将 [-1,0] 映射到 [0,1]
            recon_depth_denorm = np.clip(recon_depth_denorm, 0.0, 1.0)
            recon_depth_uint8 = (recon_depth_denorm * 1000.0).astype(np.uint8)  # 此时数值范围约为 [0,1000]
            # 新增修改：squeeze掉深度图第一个维度，使得 shape 变为 (H, W)
            recon_depth_uint8 = np.squeeze(recon_depth_uint8, axis=0)  # 现在 shape 为 (H, W)
        
            # 7) 保存RGB与Depth图像，RGB转BGR保存
            cv2.imwrite(f"{visual_dir}/view{v}_rgb_after.png", recon_rgb_uint8[..., ::-1])
            cv2.imwrite(f"{visual_dir}/view{v}_depth_after.png", recon_depth_uint8)
    else:
        # 新增逻辑：当不进行RGB与Depth拼接时，recons输出张量形状为 (1,3,3,input_height,input_width)
        recons_np = recons.cpu().numpy()  # shape (1, 3, 3, H, W) ——3通道输出仅为RGB
        recons_np = recons_np[0]  # 去除 batch 维度，结果 shape 为 (3, 3, H, W)
        
        # 新增：定义RGB反归一化的均值和标准差，形状为 (3, 1, 1)
        mean_rgb = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std_rgb  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        # 此时，第一个维度为通道 (RGB)，第二个维度为视角(T)，H, W 为图片尺寸
        for v in range(recons_np.shape[1]):
            # 直接取出每个视角的RGB图像，shape (3, H, W)
            recon_rgb = recons_np[:, v, ...]  # shape (3, H, W)
        
            # 反归一化RGB: (x * std + mean)
            recon_rgb_denorm = recon_rgb * std_rgb + mean_rgb  # shape (3, H, W)
            recon_rgb_denorm = np.clip(recon_rgb_denorm, 0.0, 1.0)
        
            # 将RGB从 (3, H, W) 转换为 (H, W, 3)
            recon_rgb_hw3 = np.transpose(recon_rgb_denorm, (1, 2, 0))  # shape (H, W, 3)
        
            # 将RGB数据乘以255并转换为 uint8 格式
            recon_rgb_uint8 = (recon_rgb_hw3 * 255).astype(np.uint8)
        
            # 保存RGB图像，转换为BGR格式保存
            cv2.imwrite(f"{visual_dir}/view{v}_rgb_after.png", recon_rgb_uint8[..., ::-1])
    
    print('done')


if __name__ == '__main__':
    sample = torch.randn(1, 4, 25, 520, 784).cuda()
    model = AutoencoderKLCogVideoX(
        in_channels=4,    # 因为我们这里是RGBD共4通道输入
        out_channels=4,   # 重建输出也包含4通道
        sample_height=520,
        sample_width=784,
        latent_channels=4,
        temporal_compression_ratio=4.0,
    ).cuda()
    with torch.no_grad():
        out = model(sample)
        middle = model.encode(sample)
    # import pdb;pdb.set_trace()
    print(out.sample.shape) # torch.Size([1, 4, 1, 65, 98])
    print(middle.latent_dist.mode().shape) # torch.Size([1, 4, 1, 65, 98])
    print('done')

    # main()