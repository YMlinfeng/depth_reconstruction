import os
import torch
import cv2
import json
import numpy as np
import torch.nn.functional as F
from model import LSSTPVDAv2Tem
from vismask.gen_vis_mask import sdf2occ


def main():

    # 帧数
    num_frames = 8      # 输入4帧，预测4帧
    interval = 5        # 间隔5帧，500ms
    num_cams = 3

    # 保存路径
    visual_dir = './pretrain/visual_dir'
    os.makedirs(visual_dir, exist_ok=True)

    # 加载信息
    infos = open('/mnt/bn/pretrain3d/real_word_data/preprocess/jsonls/dcr_data/2024_08_07-20_00_47-client_ae77079a-db7a-424d-8f6a-1f8c6a024b28.jsonl', 'r').readlines()
    samples = []
    for i in range(num_frames // 2):
        samples.append(json.loads(infos[i * interval]))
    
    # 加载前视三目图像, 内外参, slam pose (odom pose), 时间信息
    views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    imgs, intrins, extrins, ego2global, time = [], [], [], [], []
    ref_time = os.path.basename(samples[0]['images'][views[0]]).split('.')[0]
    for sample in samples:
        imgs += [cv2.imread(sample['images'][view]) for view in views]
        intrins += [sample['intrinsics'][view] for view in views]
        extrins += [sample['extrinsics'][view] for view in views]
        ego2global.append(np.array(sample['poses']['slam_pose']))
        curr_time = os.path.basename(sample['images'][views[0]]).split('.')[0]
        time.append((np.int64(curr_time) - np.int64(ref_time)) / 1000000000)
    
    # 指定预测未来帧的时间信息
    for i in range(num_frames // 2):
        time.append(time[num_frames // 2 - 1] + 0.5 * (i + 1))      # 间隔 0.5s

    # 可视化
    for i in range(len(imgs)):
        cv2.imwrite(f'{visual_dir}/{i}.png', imgs[i])

    # 数据预处理
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    ego2imgs = []
    ego2cams = []
    intrinss = []

    # 图像输入分辨率调整
    input_shape = [518, 784]
    
    # 图像resize后，对应的内参也要resize
    src_shape = imgs[0].shape[:2]
    scale_factor = np.eye(4)
    scale_factor[0, 0] *= input_shape[1] / src_shape[1]
    scale_factor[1, 1] *= input_shape[0] / src_shape[0]

    for i in range(len(imgs)):
        imgs[i] = imgs[i][:, :, [2, 1, 0]] / 255.0
        imgs[i] = np.expand_dims((imgs[i] - mean) / std, 0)
        viewpad = np.eye(4)
        viewpad[:3, :3] = np.array(intrins[i])
        ego2img = scale_factor @ viewpad @ np.linalg.inv(np.array(extrins[i]))
        ego2imgs.append(ego2img)
        ego2cams.append(np.linalg.inv(np.array(extrins[i])))
        intrinss.append(scale_factor[:3, :3] @ np.array(intrins[i]))

    imgs = torch.from_numpy(np.concatenate(imgs, 0)).permute(0, 3, 1, 2).cuda()
    imgs = F.interpolate(imgs, size=input_shape).unsqueeze(0).to(torch.float32)

    # 将参数保存在字典内
    img_meta = dict()
    img_meta['pc_range'] = [0., -10., 0., 12., 10., 4.]
    img_meta['occ_size'] = [60, 100, 20]
    img_meta['lidar2img'] = ego2imgs
    img_meta['lidar2cam'] = ego2cams
    img_meta['cam_intrinsic'] = intrinss
    img_meta['img_shape'] = [(input_shape[0], input_shape[1], 3)]
    img_meta['ego2global'] = ego2global
    img_meta['time'] = time
    img_metas = [img_meta]

    # 创建模型并加载参数
    model = LSSTPVDAv2Tem(num_classes=4, num_frames=num_frames, num_cams=num_cams)
    model.load_state_dict(torch.load('/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/depthworldv1.66/epoch_1_125k.pth', map_location='cpu')['state_dict'], strict=False)

    # # batch_size = 2
    # imgs = torch.cat([imgs, imgs], 0)
    # img_metas *= 2
    
    # 预测occupancy
    model.cuda()
    model.eval()
    with torch.no_grad():
        esdf, voxel_feat = model(imgs, img_metas)
    
    # 将预测结果保存为map
    esdf = esdf[0, 0].cpu().numpy()
    occ = sdf2occ(esdf)
    cv2.imwrite(f'{visual_dir}/esdf.png', occ.max(-1) * 255)

    for i in range(voxel_feat.shape[1]):
        esdf = voxel_feat[0, i, 0].cpu().numpy()
        occ = sdf2occ(esdf)
        cv2.imwrite(f'{visual_dir}/esdf_tem_{i}.png', occ.max(-1) * 255)

    print('done')


if __name__ == '__main__':
    main()
