import json
import cv2
import numpy as np
import pickle
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import os


class MyDataset(Dataset):
    def __init__(self):
        self.tem = 5 # 500ms/frame
        self.frames = 4 #?????
        self.data = pickle.load(open('/mnt/bn/occupancy3d/workspace/lzy/Occ3d/pkls/dcr_data_bottom_temporal_scaleddepth_paths_241213.pkl', '+rb'))['infos']
        self.params = json.load(open('/mnt/bn/occupancy3d/workspace/lzy/Occ3d/pkls/dcr_data_bottom_params.json', 'r'))
        self.data_root = 'data/dcr_data_pretrain3d'
        self.mean = np.array([0.485, 0.456, 0.406])[None, None, :]
        self.std = np.array([0.229, 0.224, 0.225])[None, None, :]
        self.input_shape = [518, 784]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if ((idx + self.tem * (self.frames - 1)) >= len(self)) or (self.data[idx]['scene_token'] != self.data[idx + self.tem * (self.frames - 1)]['scene_token']):
            idx -= self.tem * (self.frames - 1)
        
        # get data infos
        infos = []
        for i in range(self.frames):
            infos.append(self.data[idx + self.tem * i])
        
        # get images, extrinsics, intrinsics
        imgs = []
        lidar2imgs = []
        lidar2cams = []
        cam_intrinsics = []
        for i, info in enumerate(infos):
            param = self.params[info['scene_token'].split('/')[0]]
            for cam_type, cam_info in info['cams'].items():
                img = cv2.imread(os.path.join(self.data_root, info['scene_token'], cam_info['data_path']))
                ori_shape = img.shape[:2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
                imgs.append((img / 255.0  - self.mean) / self.std)

                scale_factor = np.eye(4)
                scale_factor[0, 0] *= self.input_shape[1] / ori_shape[1]
                scale_factor[1, 1] *= self.input_shape[0] / ori_shape[0]
                
                intrinsic = np.array(param['intrinsics'][cam_type])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                cam_intrinsics.append(scale_factor @ viewpad)

                lidar2cam = np.linalg.inv(np.array(param['extrinsics'][cam_type]))
                lidar2cams.append(lidar2cam)

                lidar2img = (scale_factor @ viewpad @ lidar2cam)
                lidar2imgs.append(lidar2img)
        
        source = np.asarray(imgs[:len(imgs) // 2]).astype(np.float32)
        target = np.asarray(imgs[len(imgs) // 2:]).astype(np.float32)
        
        lidar2imgs = np.asarray(lidar2imgs[:len(imgs) // 2]).astype(np.float32)
        lidar2cams = np.asarray(lidar2cams[:len(imgs) // 2]).astype(np.float32)
        cam_intrinsics = np.asarray(cam_intrinsics[:len(imgs) // 2]).astype(np.float32)

        # get x,y,yaw
        source_pose = infos[self.frames // 2 - 1]['ego2global_transformation'].astype(np.float32)
        target_pose = infos[self.frames - 1]['ego2global_transformation'].astype(np.float32)
        transform = np.linalg.inv(source_pose) @ target_pose

        yaw = Rotation.from_matrix(transform[:3, :3]).as_euler('zyx', degrees=False).tolist()[0]
        x, y = transform[0, 3], transform[1, 3]
        pose = np.array([x, y, yaw]).astype(np.float32)

        # get x, y, yaw, for test
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

        return dict(jpg=target, 
                    txt=pose[None, :], 
                    hint=source, 
                    lidar2imgs=lidar2imgs, 
                    lidar2cams=lidar2cams, 
                    cam_intrinsics=cam_intrinsics,
                    #test_pose=test_pose,
                    )




