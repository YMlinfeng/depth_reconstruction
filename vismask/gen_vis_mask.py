import numpy as np
import torch
import cc3d
import cv2


def sdf2occ(sdf: np.array):
    h, w, d = sdf.shape
    sdf[sdf > 0] = 0
    sdf[sdf < 0] = 1

    labels = cc3d.connected_components(sdf, connectivity=18)
    unique, counts = np.unique(labels, return_counts=True)

    if unique.shape != (1, ):
        max_label = unique[np.argmax(counts[1:]) + 1]
        # free为0，地面为1，障碍物为2
        sdf = np.where(labels == max_label, 1, 0)
        
    sdf[:1] = 0
    sdf[:, :1] = 0
    sdf[:, -1:] = 0
    sdf[:, :, -1:] = 0
    sdf[:, :, :1] = 0

    sdf = sdf.max(-1, keepdims=True)
    sdf = np.repeat(sdf, d, -1)
    return sdf


def voxel2points(voxel: np.array, pc_range=[0, -10, 0, 12, 10, 4.], voxel_size=0.2):
    x = torch.linspace(0, voxel.shape[0] - 1, voxel.shape[0])
    y = torch.linspace(0, voxel.shape[1] - 1, voxel.shape[1])
    z = torch.linspace(0, voxel.shape[2] - 1, voxel.shape[2])
    X, Y, Z = torch.meshgrid(x, y, z)
    xyz = torch.stack([X, Y, Z], dim=-1)

    voxel = np.expand_dims(voxel, axis=-1)
    voxel = np.concatenate([xyz.numpy(), voxel], -1)
    voxel = voxel[voxel[..., 3] > 0]

    voxel[:, :3] = (voxel[:, :3] + 0.5) * voxel_size
    voxel[:, 0] += pc_range[0]
    voxel[:, 1] += pc_range[1]
    voxel[:, 2] += pc_range[2]

    return voxel