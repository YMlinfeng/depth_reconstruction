import torch
import torch.nn as nn
import numpy as np
from modules.lsstpv_head import LSSTPVHead
# from modules.lsstpvtem_head import LSSTPVHeadTem
from modules.depth_anything_v2.dpt import DepthAnythingV2
from einops import rearrange

class LSSTPVDAv2OnlyForVoxel(nn.Module):
    def __init__(self, num_classes=1, args=None):
        super().__init__()
        self.args = args
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        encoder = "vits"  # or 'vitl', 'vitb', 'vitg'
        self.depthanythingv2 = DepthAnythingV2(**model_configs[encoder])

        '''
            LSSTPVHead 负责 3D 体素预测，输出：
                SDF（占用网格）
                Feature Map（用于进一步推理）
                Depth Map（深度图）
        '''
        self.pts_bbox_head = LSSTPVHead(
            args=self.args,
            volume_h=[60, 30, 15, 8],
            volume_w=[100, 50, 25, 13],
            volume_z=[20, 10, 5, 3],
            num_classes=num_classes,
            conv_input=[512, 256, 256, 128, 128, 128, 128],
            conv_output=[256, 256, 128, 128, 128, 128, 64],
            out_indices=[0, 2, 4, 6],
            upsample_strides=[1, 2, 1, 2, 1, 2, 1],
            pad_size=[518, 784],
            downsample=14,
            numC_input=384,
        )

    def forward(self, images, img_metas):
        '''
        pp img_metas
            [{'cam_intrinsic': [array([[415.47916667,   0.        , 395.26666667],
                [  0.        , 382.11884058, 219.96231884],
                [  0.        ,   0.        ,   1.        ]]),
                                array([[414.25416667,   0.        , 392.6125    ],
                [  0.        , 380.80507246, 226.90652174],
                [  0.        ,   0.        ,   1.        ]]),
                                array([[410.17083333,   0.        , 391.59166667],
                [  0.        , 377.05144928, 222.40217391],
                [  0.        ,   0.        ,   1.        ]])],
            'img_shape': [(518, 784, 3)],
            'lidar2cam': [array([[-0.        , -1.        , -0.        , -0.        ],
                [ 0.2079369 , -0.        , -0.97826803,  0.17532153],
                [ 0.97826803,  0.        ,  0.2079369 , -0.44939747],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                            array([[ 0.70868413, -0.70534862, -0.0151328 , -0.11716321],
                [ 0.14507774,  0.16673774, -0.9754305 ,  0.16654361],
                [ 0.69014588,  0.68925276,  0.22059373, -0.44192652],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                            array([[-0.70931232, -0.7049747 ,  0.01975357,  0.11944743],
                [ 0.14236838, -0.17061889, -0.97491786,  0.16760282],
                [ 0.69064931, -0.68874671,  0.22126643, -0.4426581 ],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])],
            'lidar2img': [array([[ 3.86676743e+02, -4.15479167e+02,  8.21905259e+01,
                    -1.77631842e+02],
                [ 2.94638712e+02,  0.00000000e+00, -3.28076362e+02,
                    -3.18568507e+01],
                [ 9.78268030e-01,  0.00000000e+00,  2.07936902e-01,
                    -4.49397475e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    1.00000000e+00]]),
                            array([[ 5.64535256e+02, -2.15843553e+01,  8.03390323e+01,
                    -2.22041226e+02],
                [ 2.11844941e+02,  2.19890524e+02, -3.21394725e+02,
                    -3.68553585e+01],
                [ 6.90145884e-01,  6.89252764e-01,  2.20593732e-01,
                    -4.41926522e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    1.00000000e+00]]),
                            array([[-2.04867125e+01, -5.58867532e+02,  9.47484271e+01,
                    -1.24347373e+02],
                [ 2.07282111e+02, -2.17510866e+02, -3.18384057e+02,
                    -3.52532373e+01],
                [ 6.90649310e-01, -6.88746713e-01,  2.21266431e-01,
                    -4.42658102e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    1.00000000e+00]])],
            'occ_size': [60, 100, 20],
            'pc_range': [0.0, -10.0, 0.0, 12.0, 10.0, 4.0]}]
        
        '''
        # (Pdb) pp img_metas[0].keys()
        # dict_keys(['pc_range', 'occ_size', 'lidar2img', 'lidar2cam', 'cam_intrinsic', 'img_shape'])
        b, n, c, h, w = images.shape # torch.Size([1, 3, 3, 518, 784])

        img_feats = self.depthanythingv2(images.view(b * n, c, h, w)) # torch.Size([3, 3, 518, 784]) 2 tuple(4, 2) #?
        img_feats = img_feats[-1][0].view(b, n, h // 14, w // 14, -1) # torch.Size([1, 3, 37, 56, 384])

        img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous() # torch.Size([1, 3, 384, 37, 56]) 把c提前了

        poses = self.get_poses(img_metas, img_feats)
        # import pdb; pdb.set_trace()

        esdf, feature, depth = self.pts_bbox_head(img_feats, poses, img_metas)

        return esdf, feature, depth
    
    def get_features(self, images, img_metas):
        b, n, c, h, w = images.shape # N为相机数量=3，为前视三目

        img_feats = self.depthanythingv2(images.view(b * n, c, h, w))
        img_feats = (
            img_feats[-1][0]
            .view(b * n, h // 14, w // 14, -1)
            .view(b, n, h // 14, w // 14, -1)
        )
        img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous()

        poses = self.get_poses(img_metas, img_feats)
        esdf, feat, depth = self.pts_bbox_head(img_feats, poses, img_metas)

        return feat
    
    def get_poses(self, img_metas, img_feats):
        poses = []
        for img_meta in img_metas:
            tmps = self.transform_pose(img_meta)
            pose = [torch.from_numpy(np.asarray(tmp)) for tmp in tmps]
            poses.append(pose)
        poses = [torch.stack(x, 0).to(img_feats) for x in zip(*poses)]

        return poses
    
    def transform_pose(self, img_meta):
        rots, trans, intrins, post_rots, post_trans, bda = [], [], [], [], [], []

        for i in range(len(img_meta['lidar2cam'])):
            cam2lidar = np.linalg.inv(img_meta['lidar2cam'][i])

            post_rot = np.eye(2)
            post_tran = np.zeros(2)
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]

            height, width, _ = img_meta['img_shape'][0]
            crop = (0, 0, width, height)
            post_rot2, post_tran2 = self.img_transform(post_rot, post_tran,
                                   resize=1.0, resize_dims=(width, height),
                                   crop=crop, flip=False, rotate=0)
            
            post_tran = np.zeros(3)
            post_rot = np.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            intrins.append(img_meta['cam_intrinsic'][i][:3, :3])
        
        bda.append(np.eye(3))

        return [rots, trans, intrins, post_rots, post_trans, bda]
    
    def img_transform(self, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):

        # post-homography transformation
        post_rot *= resize
        post_tran -= np.array(crop[:2])
        if flip:
            A = np.array([[-1, 0], [0, 1]])
            b = np.array([crop[2] - crop[0], 0])
            post_rot = A @ post_rot
            post_tran = A @ post_tran + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A @ (-b) + b
        post_rot = A @ post_rot
        post_tran = A @ post_tran + b

        return post_rot, post_tran
    
    def get_rot(self, h):
        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])



class LSSTPVDAv2(nn.Module):
    def __init__(self, num_classes=1, args=None):
        # super().__init__()
        super(LSSTPVDAv2, self).__init__() # num_classes=1 指定 用于分类的类别数（例如，1 表示二分类，4 可能表示 4 类占用情况）
        self.args = args
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        encoder = "vits"  # or 'vitl', 'vitb', 'vitg'
        self.depthanythingv2 = DepthAnythingV2(**model_configs[encoder])
        # self.depthanythingv2 = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])

        '''
            LSSTPVHead 负责 3D 体素预测，输出：
                SDF（占用网格）
                Feature Map（用于进一步推理）
                Depth Map（深度图）
        '''
        self.pts_bbox_head = LSSTPVHead(
            args=self.args,
            volume_h=[60, 30, 15, 8],
            volume_w=[100, 50, 25, 13],
            volume_z=[20, 10, 5, 3],
            num_classes=num_classes,
            conv_input=[512, 256, 256, 128, 128, 128, 128],
            conv_output=[256, 256, 128, 128, 128, 128, 64],
            out_indices=[0, 2, 4, 6],
            upsample_strides=[1, 2, 1, 2, 1, 2, 1],
            pad_size=[518, 784],
            downsample=14,
            numC_input=384,
        )

    def forward(self, images, img_metas):
        
        # (Pdb) pp img_metas[0].keys()
        # dict_keys(['pc_range', 'occ_size', 'lidar2img', 'lidar2cam', 'cam_intrinsic', 'img_shape'])
        b, n, c, h, w = images.shape # torch.Size([1, 3, 3, 518, 784])

        img_feats = self.depthanythingv2(images.view(b * n, c, h, w)) # torch.Size([3, 3, 518, 784]) 2 tuple(4, 2) #?
        img_feats = img_feats[-1][0].view(b, n, h // 14, w // 14, -1) # torch.Size([1, 3, 37, 56, 384])

        img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous() # torch.Size([1, 3, 384, 37, 56]) 把c提前了

        poses = self.get_poses(img_metas, img_feats)
        # import pdb; pdb.set_trace()

        # esdf = self.pts_bbox_head(img_feats, poses, img_metas)
        esdf, feature, depth = self.pts_bbox_head(img_feats, poses, img_metas)
        # import pdb; pdb.set_trace()
        # return esdf
        return esdf, feature, depth
    
    def get_features(self, images, img_metas):
        b, n, c, h, w = images.shape # N为相机数量=3，为前视三目

        img_feats = self.depthanythingv2(images.view(b * n, c, h, w))
        img_feats = (
            img_feats[-1][0]
            .view(b * n, h // 14, w // 14, -1)
            .view(b, n, h // 14, w // 14, -1)
        )
        img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous()

        poses = self.get_poses(img_metas, img_feats)
        esdf, feat, depth = self.pts_bbox_head(img_feats, poses, img_metas)

        return feat
    
    def get_poses(self, img_metas, img_feats):
        poses = []
        for img_meta in img_metas:
            tmps = self.transform_pose(img_meta)
            pose = [torch.from_numpy(np.asarray(tmp)) for tmp in tmps]
            poses.append(pose)
        poses = [torch.stack(x, 0).to(img_feats) for x in zip(*poses)]

        return poses
    
    def transform_pose(self, img_meta):
        rots, trans, intrins, post_rots, post_trans, bda = [], [], [], [], [], []

        for i in range(len(img_meta['lidar2cam'])):
            cam2lidar = np.linalg.inv(img_meta['lidar2cam'][i])

            post_rot = np.eye(2)
            post_tran = np.zeros(2)
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]

            height, width, _ = img_meta['img_shape'][0]
            crop = (0, 0, width, height)
            post_rot2, post_tran2 = self.img_transform(post_rot, post_tran,
                                   resize=1.0, resize_dims=(width, height),
                                   crop=crop, flip=False, rotate=0)
            
            post_tran = np.zeros(3)
            post_rot = np.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            intrins.append(img_meta['cam_intrinsic'][i][:3, :3])
        
        bda.append(np.eye(3))

        return [rots, trans, intrins, post_rots, post_trans, bda]
    
    def img_transform(self, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):

        # post-homography transformation
        post_rot *= resize
        post_tran -= np.array(crop[:2])
        if flip:
            A = np.array([[-1, 0], [0, 1]])
            b = np.array([crop[2] - crop[0], 0])
            post_rot = A @ post_rot
            post_tran = A @ post_tran + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A @ (-b) + b
        post_rot = A @ post_rot
        post_tran = A @ post_tran + b

        return post_rot, post_tran
    
    def get_rot(self, h):
        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])



'''
        pp img_metas
            [{'cam_intrinsic': [array([[415.47916667,   0.        , 395.26666667],
                [  0.        , 382.11884058, 219.96231884],
                [  0.        ,   0.        ,   1.        ]]),
                                array([[414.25416667,   0.        , 392.6125    ],
                [  0.        , 380.80507246, 226.90652174],
                [  0.        ,   0.        ,   1.        ]]),
                                array([[410.17083333,   0.        , 391.59166667],
                [  0.        , 377.05144928, 222.40217391],
                [  0.        ,   0.        ,   1.        ]])],
            'img_shape': [(518, 784, 3)],
            'lidar2cam': [array([[-0.        , -1.        , -0.        , -0.        ],
                [ 0.2079369 , -0.        , -0.97826803,  0.17532153],
                [ 0.97826803,  0.        ,  0.2079369 , -0.44939747],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                            array([[ 0.70868413, -0.70534862, -0.0151328 , -0.11716321],
                [ 0.14507774,  0.16673774, -0.9754305 ,  0.16654361],
                [ 0.69014588,  0.68925276,  0.22059373, -0.44192652],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                            array([[-0.70931232, -0.7049747 ,  0.01975357,  0.11944743],
                [ 0.14236838, -0.17061889, -0.97491786,  0.16760282],
                [ 0.69064931, -0.68874671,  0.22126643, -0.4426581 ],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])],
            'lidar2img': [array([[ 3.86676743e+02, -4.15479167e+02,  8.21905259e+01,
                    -1.77631842e+02],
                [ 2.94638712e+02,  0.00000000e+00, -3.28076362e+02,
                    -3.18568507e+01],
                [ 9.78268030e-01,  0.00000000e+00,  2.07936902e-01,
                    -4.49397475e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    1.00000000e+00]]),
                            array([[ 5.64535256e+02, -2.15843553e+01,  8.03390323e+01,
                    -2.22041226e+02],
                [ 2.11844941e+02,  2.19890524e+02, -3.21394725e+02,
                    -3.68553585e+01],
                [ 6.90145884e-01,  6.89252764e-01,  2.20593732e-01,
                    -4.41926522e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    1.00000000e+00]]),
                            array([[-2.04867125e+01, -5.58867532e+02,  9.47484271e+01,
                    -1.24347373e+02],
                [ 2.07282111e+02, -2.17510866e+02, -3.18384057e+02,
                    -3.52532373e+01],
                [ 6.90649310e-01, -6.88746713e-01,  2.21266431e-01,
                    -4.42658102e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    1.00000000e+00]])],
            'occ_size': [60, 100, 20],
            'pc_range': [0.0, -10.0, 0.0, 12.0, 10.0, 4.0]}]
        
'''