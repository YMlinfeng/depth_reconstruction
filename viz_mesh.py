import numpy as np
import mcubes
import torch
from plyfile import PlyData, PlyElement

save_dir = 'pretrain/visual_dir'
voxel = np.load(f'{save_dir}/sdf.npy')
voxel = voxel[1:, 1:-1, :-1]
vertices, triangles = mcubes.marching_cubes(voxel, 0)

xmax, xmin, ymax, ymin, zmax, zmin = 12, 0., 10, -10, 4.0, 0

vertices_ = vertices.astype(np.float32)
vertices_[:, 0] = vertices_[:, 0] / 59
vertices_[:, 1] = vertices_[:, 1] / 99
vertices_[:, 2] = vertices_[:, 2] / 19

## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
x_ = (xmax-xmin) * vertices_[:, 0] + xmin
y_ = (ymax-ymin) * vertices_[:, 1] + ymin
vertices_[:, 0] = x_
vertices_[:, 1] = y_
vertices_[:, 2] = (zmax-zmin) * vertices_[:, 2] + zmin
vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
face['vertex_indices'] = triangles

PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
            PlyElement.describe(face, 'face')]).write(f'{save_dir}/mesh.ply')

print('done')