# -*- coding: utf-8 -*-
import os
import torch
import cv2
import json
import numpy as np
import torch.nn.functional as F
from model import LSSTPVDAv2
from vismask.gen_vis_mask import sdf2occ
import matplotlib.pyplot as plt
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def get_device():
    """自动检测 CUDA 设备，如果不可用则回退到 CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    device = get_device()
    print(device)
    print(cv2.__file__)
    print(os.__file__)
    # import debugpy
    # # 监听端口
    # debugpy.listen(("127.0.0.1", 5679))
    # # 等待调试器连接（可选）
    # print("等待调试器连接...") #按F5
    # debugpy.wait_for_client()
    
    # 保存路径
    visual_dir = './output/d408/t1'
    os.makedirs(visual_dir, exist_ok=True)

    # 加载信息 JSON Lines，一种每行都是一个 JSON 对象的格式
    sample = open('/mnt/bn/pretrain3d/real_word_data/preprocess/jsonls/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec.jsonl', 'r').readlines()[0] # readlines()[0] 只读取第一个 JSON 对象（通常对应一帧数据）
    sample = json.loads(sample) # json.loads() 将字符串转换为 Python 字典（dict）
    '''
        import pprint
        pprint.pprint(sample)
        {'clip_name': 'clip_1719900610105184768',
        'extrinsics': {'CAM_FRONT': [[0.0,
                                    0.2078857421875,
                                    0.97802734375,
                                    0.403076171875],
                                    [-1.0, -0.0, 0.0, -0.0],
                                    [0.0,
                                    -0.97802734375,
                                    0.2078857421875,
                                    0.264892578125],
                                    [0.0, 0.0, 0.0, 1.0]],
                        'CAM_FRONT_LEFT': [[0.708984375,
                                            0.1451416015625,
                                            0.6904296875,
                                            0.364013671875],
                                        [-0.705078125,
                                            0.166748046875,
                                            0.68896484375,
                                            0.194091796875],
                                        [-0.01507568359375,
                                            -0.97509765625,
                                            0.220458984375,
                                            0.258056640625],
                                        [0.0, 0.0, 0.0, 1.0]],
                        'CAM_FRONT_RIGHT': [[-0.708984375,
                                            0.142333984375,
                                            0.6904296875,
                                            0.366455078125],
                                            [-0.70458984375,
                                            -0.1705322265625,
                                            -0.6884765625,
                                            -0.1920166015625],
                                            [0.019775390625,
                                            -0.97509765625,
                                            0.2213134765625,
                                            0.259033203125],
                                            [0.0, 0.0, 0.0, 1.0]]},
        'frame_id': 0,
        'images': {'CAM_FRONT': '/mnt/bn/pretrain3d/real_word_data/preprocess/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec/clip_1719900610105184768/images/bottom_front/1719900610170976768.jpg',
                    'CAM_FRONT_LEFT': '/mnt/bn/pretrain3d/real_word_data/preprocess/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec/clip_1719900610105184768/images/bottom_front_left/1719900610201669632.jpg',
                    'CAM_FRONT_RIGHT': '/mnt/bn/pretrain3d/real_word_data/preprocess/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec/clip_1719900610105184768/images/bottom_front_right/1719900610167873792.jpg'},
        'intrinsics': {'CAM_FRONT': [[1017.5, 0.0, 968.0],
                                    [0.0, 1018.0, 586.0],
                                    [0.0, 0.0, 1.0]],
                        'CAM_FRONT_LEFT': [[1014.5, 0.0, 961.5],
                                        [0.0, 1014.5, 604.5],
                                        [0.0, 0.0, 1.0]],
                        'CAM_FRONT_RIGHT': [[1004.5, 0.0, 959.0],
                                            [0.0, 1004.5, 592.5],
                                            [0.0, 0.0, 1.0]]},
        'lidar': '/mnt/bn/pretrain3d/real_word_data/preprocess/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec/clip_1719900610105184768/lidar/1719900610205353728.npy',
        'lidar_extrinsics': {'CAM_FRONT': [[0.00734710693359375,
                                            -1.0,
                                            -0.0138702392578125,
                                            -0.0323486328125],
                                            [0.23681640625,
                                            0.0152130126953125,
                                            -0.9716796875,
                                            -1.71875],
                                            [0.9716796875,
                                            0.00385284423828125,
                                            0.23681640625,
                                            -0.04638671875],
                                            [0.0, 0.0, 0.0, 1.0]],
                            'CAM_FRONT_LEFT': [[0.71435546875,
                                                -0.7001953125,
                                                -0.003787994384765625,
                                                -0.1689453125],
                                                [0.1727294921875,
                                                0.181396484375,
                                                -0.96826171875,
                                                -1.716796875],
                                                [0.67822265625,
                                                0.69091796875,
                                                0.25048828125,
                                                0.0075225830078125],
                                                [0.0, 0.0, 0.0, 1.0]],
                            'CAM_FRONT_RIGHT': [[-0.7041015625,
                                                -0.7099609375,
                                                -0.011077880859375,
                                                0.1346435546875],
                                                [0.17236328125,
                                                -0.1558837890625,
                                                -0.97265625,
                                                -1.7255859375],
                                                [0.68896484375,
                                                -0.68701171875,
                                                0.2320556640625,
                                                -0.036224365234375],
                                                [0.0, 0.0, 0.0, 1.0]]},
        'poses': {'odom_pose': [[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]],
                'slam_pose': [[0.66064453125, -0.75048828125, 0.0, 106.8125],
                                [0.75048828125, 0.66064453125, 0.0, -4.40625],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]]},
        'seq_name': '2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec'}

    
    '''
    # 加载前视三目图像
    views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    imgs = [cv2.imread(sample['images'][view]) for view in views]

    # 可视化
    for i in range(len(imgs)):
        cv2.imwrite(f'{visual_dir}/{i}.png', imgs[i])

    # 加载三个相机各自的内外参
    # intrins 存储各相机的 内参矩阵（用于像素坐标到相机坐标的变换）。
    # extrins 存储各相机的 外参矩阵（用于相机坐标到世界坐标的变换）。
    intrins = [sample['intrinsics'][view] for view in views]
    extrins = [sample['extrinsics'][view] for view in views]
    
    # 数据预处理
    mean = np.array([0.485, 0.456, 0.406])  # 归一化均值
    std = np.array([0.229, 0.224, 0.225])  # 归一化标准差
    ego2imgs = []  # 存储转换矩阵（从 ego 车体坐标系到图像坐标系）
    ego2cams = []  # 存储转换矩阵（从 ego 车体坐标系到相机坐标系）
    intrinss = []  # 存储调整后的内参矩阵

    # 图像输入分辨率调整
    input_shape = [518, 784]  # !目标输入分辨率

    # 图像resize后，对应的内参也要resize
    src_shape = imgs[0].shape[:2]  # 原始图像的高度和宽度
    scale_factor = np.eye(4)  # 4x4 单位矩阵
    scale_factor[0, 0] *= input_shape[1] / src_shape[1]  # 调整 x 方向的缩放因子
    scale_factor[1, 1] *= input_shape[0] / src_shape[0]  # 调整 y 方向的缩放因子

    for i in range(3): # 将图像归一化，并计算从车体坐标系 (ego) 到图像坐标系 (img) 和相机坐标系 (cam) 的转换矩阵
        imgs[i] = imgs[i][:, :, [2, 1, 0]] / 255.0
        imgs[i] = np.expand_dims((imgs[i] - mean) / std, 0)
        viewpad = np.eye(4)
        viewpad[:3, :3] = np.array(intrins[i])
        ego2img = scale_factor @ viewpad @ np.linalg.inv(np.array(extrins[i]))
        ego2imgs.append(ego2img)
        ego2cams.append(np.linalg.inv(np.array(extrins[i])))
        intrinss.append(scale_factor[:3, :3] @ np.array(intrins[i]))

    imgs = torch.from_numpy(np.concatenate(imgs, 0)).permute(0, 3, 1, 2).to(device) # 将 imgs 转换为 PyTorch 张量，并调整格式（(B, C, H, W)）。#!单帧
    imgs = F.interpolate(imgs, size=input_shape).unsqueeze(0).to(torch.float32) # 对 imgs 进行插值操作，将其调整为指定的 input_shape。

    # 将参数保存在字典内，img_meta 存储相机参数、点云范围等信息，供模型推理使用
    img_meta = dict()
    img_meta['pc_range'] = [0., -10., 0., 12., 10., 4.]
    img_meta['occ_size'] = [60, 100, 20]
    img_meta['lidar2img'] = ego2imgs
    img_meta['lidar2cam'] = ego2cams
    img_meta['cam_intrinsic'] = intrinss
    img_meta['img_shape'] = [(input_shape[0], input_shape[1], 3)]
    img_metas = [img_meta]

    # 创建模型并加载参数
    # ----

    model = LSSTPVDAv2(num_classes=4)
    # 下一步会 覆盖 model 之前的 随机初始化参数
    # 加载原始权重文件
    state_dict = torch.load(
        '/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth',
        map_location='cpu'
    )['state_dict']
    model.load_state_dict(state_dict, strict=False)

    loaded = torch.load(
        # '/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints/vqvae_epoch1_step12500.pth',
        "/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints_vqgan_1024_4/vqvae_epoch1_step1000.pth",
        map_location='cpu'
    )['vqvae_state_dict']
    vqvae_state = {k.replace('module.', ''): v for k, v in loaded.items()}
    print(len(vqvae_state)) # 1565

    # 再次加载，只加载筛选后的部分，这些键如果在模型中已有对应项，则会覆盖原来的值
    # model.load_state_dict(vqvae_state, strict=False)
    # print(model.load_state_dict(vqvae_state, strict=False))
    # self.vqvae = VAERes2DImgDirectBC(inp_channels=80, out_channels=80, z_channels=256, mid_channels=1024) #!!!
    model.pts_bbox_head.vqvae.load_state_dict(vqvae_state, strict=False)
    
    
    # # batch_size = 2
    # imgs = torch.cat([imgs, imgs], 0)
    # img_metas *= 2
    
    # 预测occupancy
    model.to(device) # 这行代码 将 model 从 CPU 内存移动到 GPU 显存（通常是 cuda:0）
    model.eval() # 关闭 Dropout（用于训练时随机丢弃部分神经元，使得推理结果稳定）；让 Batch Normalization（BN）层使用保存的均值和方差（而不是训练时的动态计算）；计算图 不会保存反向传播的梯度，减少显存占用
    with torch.no_grad(): ## 关闭 自动梯度计算，减少显存占用，提高推理速度
        # voxel的4通道是sdf+RGB
        # import pdb;pdb.set_trace()
        result = model(imgs, img_metas) # 返回tuple,[voxel:torch.Size([1, 4, 60, 100, 20]), voxel feature: torch.Size([1, 64, 60, 100, 20]), torch.Size([1, 304584, 1])]

    


    # .detach() 从计算图中分离张量，使其不再参与计算图（即 不会计算梯度）。
    # 如果不使用 .detach()，后续的 .cpu() 可能会报错，因为 PyTorch 仍然认为它是需要计算梯度的张量。
    # .squeeze() 去掉 shape 中的维度为 1 的维度
    # todo 这里的sdf的作用是什么?#?
    sdf = result[0][0, 0].detach().squeeze().cpu().numpy() # 最终，显存主要被 model 和 result 占用，推理结束后 result 存在 GPU，除非 .cpu() 取回，否则无法进行可视化
    costmap = sdf2occ(sdf) # sdf2occ() 将 sdf 转换为 occupancy map
    costmap = cv2.rotate(costmap.max(-1), cv2.ROTATE_180) #(60, 100)
    cv2.imwrite(f'{visual_dir}/map.png', costmap * 255)

    # save depth
    # 从(1,304584,1) 变成 (3, 259, 392)
    depth = result[-1].view(len(views), input_shape[0] // 2, input_shape[1] // 2).detach().squeeze().cpu().numpy()
    for i in range(len(views)):
        plt.imsave(f'{visual_dir}/depth_{i}.png', np.log10(depth[i]), cmap='jet')
    print('done')


if __name__ == '__main__':
    main()

