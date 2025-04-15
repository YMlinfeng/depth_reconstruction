import os
import torch
import cv2
import json
import numpy as np
import torch.nn.functional as F
from model import LSSTPVDAv2
from vismask.gen_vis_mask import sdf2occ
import matplotlib.pyplot as plt


def main():

    # 保存路径
    visual_dir = './visual_dir'
    os.makedirs(visual_dir, exist_ok=True)

    # 加载信息
    sample = open('/mnt/bn/pretrain3d/real_word_data/preprocess/jsonls/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec.jsonl', 'r').readlines()[0]
    sample = json.loads(sample)
    
    # 加载前视三目图像
    views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    imgs = [cv2.imread(sample['images'][view]) for view in views]

    # 可视化
    for i in range(len(imgs)):
        cv2.imwrite(f'{visual_dir}/{i}.png', imgs[i])

    # 加载三个相机各自的内外参
    intrins = [sample['intrinsics'][view] for view in views]
    extrins = [sample['extrinsics'][view] for view in views]
    
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

    for i in range(3):
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
    img_metas = [img_meta]

    # 创建模型并加载参数
    model = LSSTPVDAv2(num_classes=4)
    model.load_state_dict(torch.load('/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth', map_location='cpu')['state_dict'], strict=False)

    # # batch_size = 2
    # imgs = torch.cat([imgs, imgs], 0)
    # img_metas *= 2
    
    # 预测occupancy
    model.cuda()
    model.eval()
    with torch.no_grad():
        result = model(imgs, img_metas)

    sdf = result[0][0, 0].detach().squeeze().cpu().numpy()
    costmap = sdf2occ(sdf)
    costmap = cv2.rotate(costmap.max(-1), cv2.ROTATE_180)
    cv2.imwrite(f'{visual_dir}/map.png', costmap * 255)

    # save depth
    depth = result[-1].view(len(views), input_shape[0] // 2, input_shape[1] // 2).detach().squeeze().cpu().numpy()
    for i in range(len(views)):
        plt.imsave(f'{visual_dir}/depth_{i}.png', np.log10(depth[i]), cmap='jet')
    print('done')


if __name__ == '__main__':
    main()
