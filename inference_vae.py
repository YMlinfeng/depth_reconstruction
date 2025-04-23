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
    parser.add_argument("--visual_dir", type=str, default='./output/d421/vae/',
                        help="保存可视化结果的输出路径")
    parser.add_argument("--jsonl_file", type=str,
                        default='/mnt/bn/pretrain3d/real_word_data/preprocess/jsonls/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec.jsonl',
                        help="包含样本信息的 JSON Lines 文件路径，读取第一行数据")
    parser.add_argument("--vqvae_weight", type=str,
                        default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints_vae_r0/vqvae_epoch8_step18000.pth",
                        help="3D Casual VAE（VQ-VAE）模型权重文件路径")
    parser.add_argument("--input_height", type=int, default=520,
                        help="图像输入高度")
    parser.add_argument("--input_width", type=int, default=784,
                        help="图像输入宽度")
    parser.add_argument("--model", type=str, default="Cog3DVAE", help="choices: VAERes2DImgDirectBC, VQModel, Cog3DVAE")
    parser.add_argument("--mode", type=str, default="eval", help="choices: train, eval")
    parser.add_argument("--general_mode", type=str, default="vae", help="choices: vae, vqgan")
    args = parser.parse_args()

    device = get_device()
    print(cv2.__file__)
    print(os.__file__)
    print(f"args: {args}")

    # 保存路径
    visual_dir = args.visual_dir
    os.makedirs(visual_dir, exist_ok=True)

    sample = open(args.jsonl_file, 'r').readlines()[0] # readlines()[0] 只读取第一个 JSON 对象（通常对应一帧数据）
    sample = json.loads(sample) # json.loads() 将字符串转换为 Python 字典（dict）

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
        if len(depth_vis.shape) == 2:
            # 归一化深度，仅示例
            depth_norm = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
            depth_vis_uint8 = (depth_norm*255).astype(np.uint8)
            cv2.imwrite(f'{visual_dir}/{i}_depth_before.png', depth_vis_uint8)
        else:
            # 若不是单通道可根据自己需求做处理
            cv2.imwrite(f'{visual_dir}/{i}_depth_before.png', depth_vis)

    # 数据预处理 参考dataset和主训练代码
    # ------
    # 这里我们模拟Dataset中的预处理逻辑：把RGB转成float32并除255.0 -> 转成RGB(若需要) -> 减均值除标准差 -> 与Depth拼接 -> resize等
    # ------
    processed_list = []
    for i in range(len(imgs)):
        # a) 先检查RGB是否成功读取
        if imgs[i] is None:
            raise ValueError(f"RGB图像{i}为空！")
        rgb_bgr = imgs[i].astype(np.float32) / 255.0  # shape (H, W, 3)
        rgb = rgb_bgr[..., [2,1,0]]  # 转成RGB通道顺序, shape仍然 (H, W, 3)

        # b) 读取Depth
        depth_data = depths[i]
        if depth_data is None:
            raise ValueError(f"深度图{i}为空！")
        # 若深度图是单通道 shape (H, W)，则扩展为(H, W, 1)
        if len(depth_data.shape) == 2:
            depth_data = np.expand_dims(depth_data, axis=-1).astype(np.float32)
        elif depth_data.shape[-1] > 1:
            depth_data = depth_data[..., :1].astype(np.float32)

        # c) 对Depth也归一化到[0,1]，假设它原本是[0,255]量程
        depth_data /= 255.0

        # d) 拼接RGB+D => (H, W, 4)
        rgbd = np.concatenate([rgb, depth_data], axis=-1)  # shape (H, W, 4)

        # e) 与训练一致，需要先减均值除以标准差(针对前3个通道RGB), 参考Dataset中的mean, std
        means = np.array([0.485, 0.456, 0.406, 0.0], dtype=np.float32).reshape(1,1,4)
        stds  = np.array([0.229, 0.224, 0.225, 1.0], dtype=np.float32).reshape(1,1,4)
        rgbd_norm = (rgbd - means) / stds  # shape还是(H, W, 4)

        # f) resize到(args.input_height, args.input_width)
        rgbd_resized = cv2.resize(
            rgbd_norm,
            (args.input_width, args.input_height),  # (宽, 高)
            interpolation=cv2.INTER_LINEAR
        )  # shape (input_height, input_width, 4)

        # g) 将 np.transpose: (H,W,C) => (C,H,W)
        rgbd_chw = np.transpose(rgbd_resized, (2,0,1))  # shape (4, input_height, input_width)

        processed_list.append(rgbd_chw)

    # h) 将三帧(或三视角)拼到一起 => shape (3, 4, H, W)
    #   这里之前是把“视角”当做batch维度 => (B=3, C=4, H, W)
    #   但训练时，我们其实想把这 3 个视角当做 “时间帧 T=3”，并让 batch=1 (B=1)
    processed_np = np.stack(processed_list, axis=0)  # shape (3, 4, H, W)

    # i) （原注释）再加一个时间frames维度 => shape (3, 4, 1, H, W)
    #    但现在根据训练需求，需要变成 [B=1, C=4, T=3, H, W]
    #    所以我们不按原先方式在 axis=2 上插入；而是先转置一下，让维度对上 (4,3,H,W)，然后在最前面插一个 batch=1
    #
    #    #todo 下面是新的处理，保证和训练代码 (1,4,3,H,W) 对齐
    #    注意：此时 processed_np 是 (3,4,H,W)，
    #         我们要把它变成 (4,3,H,W) => 再变成 (1,4,3,H,W)

    processed_np = processed_np.transpose(1, 0, 2, 3)  # => shape (4, 3, H, W)
    processed_np = np.expand_dims(processed_np, axis=0) # => shape (1, 4, 3, H, W)

    # j) 现在最终送入模型的tensor是 (1,4,3,520,784)，
    #    其中 B=1, C=4(RGBD), T=3(视角当做时间帧)，H=520, W=784
    input_tensor = torch.from_numpy(processed_np).float().to(device)

    #---【张量形状说明】
    # step h) processed_list里每个元素是(4, H, W)，stack后 => (3, 4, H, W)
    # step i) 经过 transpose(1,0,2,3) => (4,3,H,W)，然后 expand_dims => (1,4,3,H,W)
    #         这样就跟训练代码中 (B, C, T, H, W) 一致，之前 T=3 就是这里的3，batch=1。


    #todo 进行推理和重建
    # 1) 加载3D Casual VAE模型 + 加载checkpoint
    vqvae = AutoencoderKLCogVideoX(
        in_channels=4,    # 因为我们这里是RGBD共4通道输入
        out_channels=4,   # 重建输出也包含4通道
        sample_height=args.input_height,
        sample_width=args.input_width,
        latent_channels=4,
        temporal_compression_ratio=4.0,
        # 其余超参如果与你训练的配置不同，需要自行改成一致
    )
    # 2) 加载checkpoint (下面一行很关键)
    checkpoint = torch.load(args.vqvae_weight, map_location='cpu')  
    vqvae.load_state_dict(checkpoint['vqvae_state_dict'], strict=True)
    vqvae.eval()
    vqvae.to(device)

    # 3) 前向推理
    with torch.no_grad():
        encode_out = vqvae.encode(input_tensor, return_dict=True)  # =>返回AutoencoderKLOutput
        latents = encode_out.latent_dist.mode()  # shape [B, latentC, T', H', W'] = [B, 4, 1, 65, 98]
        decode_out = vqvae.decode(latents, return_dict=True)
        recons = decode_out.sample  # shape [B, 4, T, H, W], 与输入通道对应(RGBD) #todo实际上通过debug得到的是（1，4，1，520，784）这里有错误

    #可视化重建之后的图像和depth
    # 这里 recons.shape = (1, 4, 3, input_height, input_width)
    #   其中 B=1, C=4(RGB+D)，T=3(这里的3个帧对应3个视角)
    recons_np = recons.cpu().numpy()  # 转到CPU做可视化

    # 为了和之前“可视化时按视角分图”的习惯一致，这里可以把 T=3 的那一维度拆开
    # recons_np.shape => (1,4,3,H,W)
    # 拿到单个 batch => shape (4,3,H,W)
    recons_np = recons_np[0]  # => shape (4, 3, H, W) #todo 实际上通过debug这里得到的是（4，1，520，784）

    # 每个视角 v => recons_np[:, v, ...], 其中 v=0,1,2
    for v in range(recons_np.shape[1]):
        # 1) shape(4,H,W) : recons_np[:, v, ...] => (4, H, W)
        recon_i = recons_np[:, v, ...]  # => shape (4, H, W)

        # 2) 分离RGB与Depth => (3, H, W) / (1, H, W)
        recon_rgb = recon_i[:3, ...]   # => shape (3, H, W) #todo 通过debug这里张量不是全零
        recon_depth = recon_i[3:4, ...] # => shape (1, H, W) #todo 通过debug这里张量不是全零

        # 3) 把RGB转回 (H, W, 3)
        recon_rgb_hw3 = np.transpose(recon_rgb, (1,2,0))  # => (H, W, 3) #todo 通过debug这里张量变为全部都是零，发生错误！！！！

        # 4) 先把RGB限制到[0,1]之间，再乘255
        recon_rgb_hw3 = np.clip(recon_rgb_hw3, 0.0, 1.0)
        recon_rgb_uint8 = (recon_rgb_hw3 * 255).astype(np.uint8)

        # 5) Depth同理 => (H, W, 1)
        recon_depth_hw1 = np.transpose(recon_depth, (1,2,0))
        recon_depth_hw1 = np.clip(recon_depth_hw1, 0.0, 1.0)
        recon_depth_uint8 = (recon_depth_hw1*1000.0).astype(np.uint8)

        # 6) 保存
        cv2.imwrite(f"{visual_dir}/view{v}_rgb_after.png", recon_rgb_uint8[..., ::-1])  # 如需要BGR再转[..., ::-1]
        cv2.imwrite(f"{visual_dir}/view{v}_depth_after.png", recon_depth_uint8)

    print('done')


if __name__ == '__main__':
    # sample = torch.randn(1, 4, 3, 520, 784).cuda()
    # model = AutoencoderKLCogVideoX(
    #     in_channels=4,    # 因为我们这里是RGBD共4通道输入
    #     out_channels=4,   # 重建输出也包含4通道
    #     sample_height=520,
    #     sample_width=784,
    #     latent_channels=4,
    #     temporal_compression_ratio=4.0,
    #     # 其余超参如果与你训练的配置不同，需要自行改成一致
    # ).cuda()
    # with torch.no_grad():
    #     out = model(sample)
    #     # out = model.encode(sample)
    # print(out.sample.shape)
    # # print(out.latent_dist.mode().shape)
    # print('done')
    main()