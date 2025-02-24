import torch
import torch.nn as nn
import numpy as np


def grid_sample_3d(feature, grid):
    N, C, ID, IH, IW = feature.shape
    _, D, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]
    iz = grid[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    feature = feature.view(N, C, ID * IH * IW)

    max_value = ID * IH * IW - 1

    tnw_val = torch.gather(feature, 2, torch.clamp((iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1), 0, max_value))
    tne_val = torch.gather(feature, 2, torch.clamp((iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1), 0, max_value))
    tsw_val = torch.gather(feature, 2, torch.clamp((iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1), 0, max_value))
    tse_val = torch.gather(feature, 2, torch.clamp((iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1), 0, max_value))
    bnw_val = torch.gather(feature, 2, torch.clamp((iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1), 0, max_value))
    bne_val = torch.gather(feature, 2, torch.clamp((iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1), 0, max_value))
    bsw_val = torch.gather(feature, 2, torch.clamp((iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1), 0, max_value))
    bse_val = torch.gather(feature, 2, torch.clamp((iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1), 0, max_value))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val


class RaySampler(nn.Module):

    def __init__(
            self,
            ray_sample_mode='fixed',    # fixed, cellular
            ray_number=[192, 400],      # 192 * 400
            ray_img_size=[768, 1600],
            ray_upper_crop=0,
            ray_x_dsr_max=None,
            ray_y_dsr_max=None):
        super().__init__()

        self.ray_sample_mode = ray_sample_mode
        self.ray_number = ray_number[0] * ray_number[1]
        self.ray_resize = ray_number
        self.ray_img_size = ray_img_size
        assert ray_sample_mode in ['fixed', 'cellular', 'random'] # TODO

        if ray_sample_mode == 'fixed':
            ray_x_dsr = 1.0 * ray_img_size[1] / ray_number[1]
            ray_y_dsr = 1.0 * ray_img_size[0] / ray_number[0]
            ray_x = torch.arange(ray_number[1], dtype=torch.float) * ray_x_dsr
            ray_y = torch.arange(ray_number[0], dtype=torch.float) * ray_y_dsr
            rays = torch.stack([
                ray_x.unsqueeze(0).expand(ray_number[0], -1),
                ray_y.unsqueeze(1).expand(-1, ray_number[1])], dim=-1).flatten(0, 1) # HW, 2
            self.register_buffer('rays', rays, False)
        elif ray_sample_mode == 'cellular':
            self.ray_upper_crop = ray_upper_crop
            self.ray_x_dsr_max = 1.0 * ray_img_size[1] / ray_number[1]
            self.ray_y_dsr_max = 1.0 * (ray_img_size[0] - ray_upper_crop) / ray_number[0]
            if ray_x_dsr_max is not None:
                self.ray_x_dsr_max = ray_x_dsr_max
            if ray_y_dsr_max is not None:
                self.ray_y_dsr_max = ray_y_dsr_max
            assert self.ray_x_dsr_max > 1 and self.ray_y_dsr_max > 1
            ray_x = torch.arange(ray_number[1], dtype=torch.float)
            ray_y = torch.arange(ray_number[0], dtype=torch.float)
            rays = torch.stack([
                ray_x.unsqueeze(0).expand(ray_number[0], -1),
                ray_y.unsqueeze(1).expand(-1, ray_number[1])], dim=-1) # H, W, 2
            self.register_buffer('rays', rays, False)

    def forward(self):
        device = self.rays.device
        
        if self.ray_sample_mode == 'fixed':
            return self.rays
        elif self.ray_sample_mode == 'random':
            rays = torch.rand(self.ray_number, 2, device=device)
            rays[:, 0] = rays[:, 0] * self.ray_img_size[1]
            rays[:, 1] = rays[:, 1] * self.ray_img_size[0]
            return rays
        elif self.ray_sample_mode == 'cellular':
            ray_x_dsr = np.random.uniform() * (self.ray_x_dsr_max - 1) + 1
            ray_y_dsr = np.random.uniform() * (self.ray_y_dsr_max - 1) + 1
            ray_x_emp_max = self.ray_img_size[1] - self.ray_resize[1] * ray_x_dsr
            ray_y_emp_max = self.ray_img_size[0] - self.ray_upper_crop - self.ray_resize[0] * ray_y_dsr
            ray_x_emp = np.random.uniform() * ray_x_emp_max
            ray_y_emp = np.random.uniform() * ray_y_emp_max
            rays = self.rays.clone() # H, W, 2
            rays[..., 0] = rays[..., 0] * ray_x_dsr + ray_x_emp
            rays[..., 1] = rays[..., 1] * ray_y_dsr + ray_y_emp + self.ray_upper_crop
            return rays.flatten(0, 1)
