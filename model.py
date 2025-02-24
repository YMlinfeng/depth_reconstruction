import torch
import torch.nn as nn
import numpy as np
from modules.lsstpv_head import LSSTPVHead
from modules.lsstpvtem_head import LSSTPVHeadTem
from modules.depth_anything_v2.dpt import DepthAnythingV2
from einops import rearrange


class LSSTPVDAv2(nn.Module):
    def __init__(self, num_classes=1):
        super(LSSTPVDAv2, self).__init__()

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
        self.depthanythingv2 = DepthAnythingV2(**{**model_configs[encoder]})

        self.pts_bbox_head = LSSTPVHead(
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
        b, n, c, h, w = images.shape

        img_feats = self.depthanythingv2(images.view(b * n, c, h, w))
        img_feats = (
            img_feats[-1][0]
            .view(b * n, h // 14, w // 14, -1)
            .view(b, n, h // 14, w // 14, -1)
        )
        img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous()

        poses = self.get_poses(img_metas, img_feats)

        esdf, feat, depth = self.pts_bbox_head(img_feats, poses, img_metas)

        return esdf, feat, depth
    
    def get_features(self, images, img_metas):
        b, n, c, h, w = images.shape

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


class LSSTPVDAv2Tem(nn.Module):
    def __init__(self, num_classes=1, num_frames=8, num_cams=3):
        super(LSSTPVDAv2Tem, self).__init__()

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
        self.depthanythingv2 = DepthAnythingV2(**{**model_configs[encoder]})

        self.pts_bbox_head = LSSTPVHeadTem(
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
            num_frames=num_frames,
            num_imgs=num_cams,
        )

        self.num_frames = num_frames
        self.num_cams = num_cams

    def forward(self, images, img_metas):
        b, n, c, h, w = images.shape

        img_feats = self.depthanythingv2(images.view(b * n, c, h, w))
        img_feats = (
            img_feats[-1][0]
            .view(b * n, h // 14, w // 14, -1)
            .view(b, n, h // 14, w // 14, -1)
        )
        img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous()
        img_feats = rearrange(img_feats, "b (f n) c h w -> (b f) n c h w", f=self.num_frames // 2)

        poses = self.get_poses(img_metas, img_feats)

        esdf, voxel_feat = self.pts_bbox_head(img_feats, poses, img_metas)

        return esdf, voxel_feat
    
    def get_poses(self, img_metas, img_feats):
        poses = []
        for img_meta in img_metas:
            tmps = self.transform_pose(img_meta)
            pose = [torch.from_numpy(np.asarray(tmp)) for tmp in tmps]
            poses.append(pose)
        poses = [torch.stack(x, 0).to(img_feats) for x in zip(*poses)]

        for i in range(len(poses)):
            if i == (len(poses) - 1):
                poses[i] = poses[i].repeat(self.num_frames // 2, 1, 1, 1)
            else:
                shape = poses[i].shape
                poses[i] = poses[i].view(shape[0] * self.num_frames // 2, self.num_cams, *shape[2:])

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