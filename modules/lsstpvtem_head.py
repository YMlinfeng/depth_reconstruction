import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from modules.occformer.occnet import OccupancyEncoder
from modules.image2bev.ViewTransformerLSSVoxel import (
    ViewTransformerLiftSplatShootVoxel,
)
from modules.cam3dbev.pred_block_dit import PredictorDiTtime
from modules.cam3dbev.fpn2d import FPN2D
from modules.cam3dbev.resnet2d import CustomResNet2D
from einops import rearrange


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class LSSTPVHeadTem(nn.Module):
    def __init__(
        self,
        num_classes=17,
        volume_h=200,
        volume_w=200,
        volume_z=16,
        upsample_strides=[1, 2, 1, 2],
        out_indices=[0, 2, 4, 6],
        conv_input=None,
        conv_output=None,
        pad_size=[736, 1280],
        downsample=16,
        numC_input=512,
        numC_Trans=128,
        num_imgs=3,
        num_frames=8,
    ):
        super(LSSTPVHeadTem, self).__init__()

        point_cloud_range = [0., -10., 0., 12., 10., 4.]
        voxel_size = 0.2
        lss_downsample = [1, 1, 1]
        data_config = {"pad_size": pad_size}
        grid_config = {
            "xbound": [
                point_cloud_range[0],
                point_cloud_range[3],
                voxel_size * lss_downsample[0],
            ],
            "ybound": [
                point_cloud_range[1],
                point_cloud_range[4],
                voxel_size * lss_downsample[1],
            ],
            "zbound": [
                point_cloud_range[2],
                point_cloud_range[5],
                voxel_size * lss_downsample[2],
            ],
            "dbound": [0.05, 10, 0.05],
        }

        self.img_view_transformer = ViewTransformerLiftSplatShootVoxel(
            grid_config=grid_config,
            data_config=data_config,
            numC_Trans=numC_Trans,
            vp_megvii=False,
            downsample=downsample,
            numC_input=numC_input,
        )

        self.img_bev_encoder_backbone_xy = OccupancyEncoder(
            num_stage=4,
            in_channels=numC_Trans,
            block_numbers=[2, 2, 2, 2],
            block_inplanes=[numC_Trans, 128, 256, 512],
            block_strides=[1, 2, 2, 2],
            out_indices=(0, 1, 2, 3),
        )

        self.img_bev_encoder_backbone_yz = OccupancyEncoder(
            num_stage=4,
            in_channels=numC_Trans,
            block_numbers=[2, 2, 2, 2],
            block_inplanes=[numC_Trans, 128, 256, 512],
            block_strides=[1, 2, 2, 2],
            out_indices=(0, 1, 2, 3),
        )

        self.img_bev_encoder_backbone_xz = OccupancyEncoder(
            num_stage=4,
            in_channels=numC_Trans,
            block_numbers=[2, 2, 2, 2],
            block_inplanes=[numC_Trans, 128, 256, 512],
            block_strides=[1, 2, 2, 2],
            out_indices=(0, 1, 2, 3),
        )

        self.numC_Trans = numC_Trans
        self.conv_input = conv_input
        self.conv_output = conv_output

        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z

        self.upsample_strides = upsample_strides
        self.out_indices = out_indices
        
        self.num_imgs = num_imgs
        self.num_frames = num_frames
        self._init_layers()

        # 定义depthworld
        time_receptive_field = self.num_frames // 2
        n_future_frames_plus = self.num_frames // 2
        numC_Trans = 80

        self.occ_encoder_backbone = CustomResNet2D(34,
                                        n_input_channels=numC_Trans * time_receptive_field,
                                        block_inplanes=[numC_Trans * time_receptive_field * i for i in [1, 2, 4, 6]],
                                        out_indices=(0, 1, 2, 3),
                                        )
        
        self.occ_predictor = PredictorDiTtime(n_input_channels=[numC_Trans * i for i in [1, 2, 4, 6]],
                                    in_timesteps=time_receptive_field,
                                    out_timesteps=n_future_frames_plus,
                                    )
        
        self.occ_encoder_neck = FPN2D(with_cp=False,
                                    in_channels=[numC_Trans * n_future_frames_plus * i for i in [1, 2, 4, 6]],
                                    out_channels=numC_Trans * n_future_frames_plus,
                                    )
        
        self.new_pc_range = [0, -12, 0, 16, 12, 4.0]
        self.new_occ_size = [80, 120, 20]

        self.spatial_extent3d = (self.new_pc_range[3] - self.new_pc_range[0], \
                                    self.new_pc_range[4] - self.new_pc_range[1], \
                                    self.new_pc_range[5] - self.new_pc_range[2])

        self.ego_center_shift_proportion_x = abs(self.new_pc_range[0]) / (self.new_pc_range[3] - self.new_pc_range[0])
        self.ego_center_shift_proportion_y = abs(self.new_pc_range[1]) / (self.new_pc_range[4] - self.new_pc_range[1])
        self.ego_center_shift_proportion_z = abs(self.new_pc_range[2]) / (self.new_pc_range[5] - self.new_pc_range[2])


    def _init_layers(
        self,
    ):
        self.deblocks = nn.ModuleList()
        upsample_strides = self.upsample_strides

        out_channels = self.conv_output
        in_channels = self.conv_input

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = nn.ConvTranspose3d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False,
                )
            else:
                upsample_layer = nn.Conv3d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )

            deblock = nn.Sequential(
                upsample_layer, nn.BatchNorm3d(out_channel), nn.ReLU(inplace=True)
            )

            self.deblocks.append(deblock)

        self.sdf = nn.ModuleList()
        for i in self.out_indices:
            sdf = nn.Conv3d(
                in_channels=out_channels[i],
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.sdf.append(sdf)

        self.combine_xy = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)
        self.combine_yz = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)
        self.combine_xz = nn.Conv3d(self.numC_Trans, 1, kernel_size=1, bias=True)


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, "conv_offset"):
                constant_init(m.conv_offset, 0)


    def forward(self, image_feats, poses, img_metas, return_features=False):
        # get tpv features based on lss
        mlp_input = self.img_view_transformer.get_mlp_input(*poses)
        geo_inputs = [*poses, mlp_input]
        xyz_feats, depth = self.img_view_transformer([image_feats] + geo_inputs)

        xy_feats = xyz_feats.mean(dim=4, keepdim=True)
        yz_feats = xyz_feats.mean(dim=2, keepdim=True)
        xz_feats = xyz_feats.mean(dim=3, keepdim=True)

        # optimize tpv features separately
        xy_feats = self.img_bev_encoder_backbone_xy(xy_feats)
        yz_feats = self.img_bev_encoder_backbone_yz(yz_feats.permute(0, 1, 3, 4, 2))
        xz_feats = self.img_bev_encoder_backbone_xz(xz_feats.permute(0, 1, 2, 4, 3))

        # fuse tpv features
        volume_embed_reshape = []
        for i in range(len(xy_feats)):
            tpv_xy = (
                xy_feats[i]
                .permute(0, 1, 2, 3, 4)
                .expand(-1, -1, -1, -1, self.volume_z[i])
            )
            tpv_yz = (
                yz_feats[i]
                .permute(0, 1, 4, 2, 3)
                .expand(-1, -1, self.volume_h[i], -1, -1)
            )
            tpv_xz = (
                xz_feats[i]
                .permute(0, 1, 2, 4, 3)
                .expand(-1, -1, -1, self.volume_w[i], -1)
            )
            xy_coeff = F.softmax(self.combine_xy(xyz_feats), dim=4)
            yz_coeff = F.softmax(self.combine_yz(xyz_feats), dim=2)
            xz_coeff = F.softmax(self.combine_xz(xyz_feats), dim=3)
            if i == 0:
                fused = (
                    tpv_xy * xy_coeff
                    + tpv_yz * yz_coeff
                    + tpv_xz * xz_coeff
                    + xyz_feats
                )
            else:
                fused = tpv_xy * xy_coeff + tpv_yz * yz_coeff + tpv_xz * xz_coeff

            scale_factor = 0.5
            output_size = [math.ceil(dim * scale_factor) for dim in xyz_feats.shape[2:]]
            xyz_feats = F.interpolate(xyz_feats, size=output_size, mode="trilinear")
            volume_embed_reshape.append(fused)

        # obtain 3d features
        outputs = []
        result = volume_embed_reshape.pop()
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)
            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks) - 1:  # we do not add skip connection at level 0
                volume_embed_temp = volume_embed_reshape.pop()
                if result.shape != volume_embed_temp.shape:
                    result = F.interpolate(
                        result, size=volume_embed_temp.shape[2:], mode="trilinear"
                    )
                result = result + volume_embed_temp
        
        # decode occupancy from 3d features
        sdf_preds = []
        for i in range(len(outputs)):
            sdf_pred = self.sdf[i](F.interpolate(outputs[i], size=outputs[-1].shape[2:], mode='trilinear'))
            sdf_preds.append(sdf_pred.contiguous())
        
        # prepare the input voxel for depthworld, 统一坐标系
        esdf = rearrange(sdf_preds[-1], '(b n) c h w d -> b n c h w d', b=len(img_metas))
        input_voxel = torch.zeros(list(esdf.shape[:3]) + self.new_occ_size).to(esdf)
        input_voxel[:, :, :, :-20, 10:-10, :] = esdf
        input_voxel = self.warp_voxels(input_voxel, img_metas)

        # prepare time infomation
        times = []
        for img_meta in img_metas:
            times.append(img_meta["time"])
        times = torch.from_numpy(np.asarray(times)).to(input_voxel)
        times = times[:, :, None, None, None, None].repeat(1, 1, self.new_occ_size[0], self.new_occ_size[1], self.new_occ_size[2], 1)

        # 计算world model
        b, t, c, h, w, d = input_voxel.shape
        inp = rearrange(input_voxel, 'b t c h w d -> b (t c d) h w')
        x = self.occ_encoder_backbone(inp)
        x = self.occ_predictor(x, times)
        x = self.occ_encoder_neck(x)
        output_voxel = 0.
        for feat in x:
            feat = F.interpolate(feat, size=[self.new_occ_size[0], self.new_occ_size[1]], mode='bilinear')
            output_voxel += feat
        output_voxel = rearrange(output_voxel, 'b (t c d) h w -> b t c h w d', t=t, c=c, d=d)

        # voxel feat, b * t * c * h * w * d
        voxel_feat = torch.cat([input_voxel, output_voxel], 1)

        return esdf[:, -1], voxel_feat


    def warp_voxels(self, voxels, img_metas):
        b, t, c, h, w, d = voxels.shape
        for bs in range(b):
            for ts in range(1, t):
                tag_pose = img_metas[bs]['ego2global'][0]
                src_pose = img_metas[bs]['ego2global'][ts]
                flow = np.linalg.inv(tag_pose) @ src_pose
                flow = torch.from_numpy(flow).unsqueeze(0).to(voxels)
                voxels[bs:bs+1, ts] = self.warp_features(voxels[bs:bs+1, ts], flow)
        return voxels
    

    def warp_features(self, x, flow):
        '''
        Warp features by motion flow
        x: b, c, h, w, d
        flow: b, 4, 4
        '''

        if flow is None:
            return x

        b, dc, dx, dy, dz = x.shape

        # normalize 3D motion flow
        flow[:,0,-1] =flow[:,0,-1]*dx/self.spatial_extent3d[0]
        flow[:,1,-1] =flow[:,1,-1]*dy/self.spatial_extent3d[1]
        flow[:,2,-1] =flow[:,2,-1]*dz/self.spatial_extent3d[2]

        nx, ny, nz = torch.meshgrid(torch.arange(dx, dtype=torch.float, device=x.device), \
                                    torch.arange(dy, dtype=torch.float, device=x.device), \
                                    torch.arange(dz, dtype=torch.float, device=x.device))
        tmp = torch.ones((dx, dy, dz), device=x.device)
        grid = torch.stack((nx, ny, nz, tmp), dim=-1)

        # centralize shift
        shift_x = self.ego_center_shift_proportion_x * dx
        shift_y = self.ego_center_shift_proportion_y * dy
        shift_z = self.ego_center_shift_proportion_z * dz

        grid[:, :, :, 0] = grid[:, :, :, 0] - shift_x
        grid[:, :, :, 1] = grid[:, :, :, 1] - shift_y
        grid[:, :, :, 2] = grid[:, :, :, 2] - shift_z
        grid = grid.view(dx*dy*dz, grid.shape[-1]).unsqueeze(-1)   #[N,4,1] 

        transformation = flow.unsqueeze(1)  # [bs, 1, 4, 4]
        transformed_grid = transformation @ grid  # [bs, N, 4, 1]
        transformed_grid = transformed_grid.squeeze(-1) # [bs, N, 4]
        transformed_grid = transformed_grid.view(-1, 4)

        # de-centralize
        transformed_grid[:, 0] = (transformed_grid[:, 0] + shift_x)
        transformed_grid[:, 1] = (transformed_grid[:, 1] + shift_y)
        transformed_grid[:, 2] = (transformed_grid[:, 2] + shift_z)
        transformed_grid = transformed_grid.round().long()

        # de-normalize
        grid = grid.squeeze(-1)
        grid = grid.view(-1, 4)
        grid[:, 0] = (grid[:, 0] + shift_x)
        grid[:, 1] = (grid[:, 1] + shift_y)
        grid[:, 2] = (grid[:, 2] + shift_z)
        grid = grid.round().long()

        batch_ix = torch.cat([torch.full([transformed_grid.shape[0] // b, 1], ix, device=x.device, dtype=torch.long) for ix in range(b)])

        kept = (transformed_grid[:,0] >= 0) & (transformed_grid[:,0] <dx) \
                & (transformed_grid[:,1] >= 0) & (transformed_grid[:,1] <dy) \
                & (transformed_grid[:,2] >= 0) & (transformed_grid[:,2] < dz)

        transformed_grid = transformed_grid[kept]
        batch_ix = batch_ix[kept]
        grid = grid[kept]

        warped_x =  torch.zeros_like(x, device=x.device) 

        # hard coding for reducing memory usage
        # erratum for new version
        split_num = 32
        gap = transformed_grid.shape[0]//split_num
        for tt in range(split_num-1):
            start_idx_tt = int(tt*gap)
            end_idx_tt = int((tt+1)*gap)
            current_batch = batch_ix[start_idx_tt:end_idx_tt]
            ixx = transformed_grid[start_idx_tt:end_idx_tt, 0]
            ixy = transformed_grid[start_idx_tt:end_idx_tt, 1]
            ixz = transformed_grid[start_idx_tt:end_idx_tt, 2]

            ixx_ori = grid[start_idx_tt:end_idx_tt, 0]
            ixy_ori = grid[start_idx_tt:end_idx_tt, 1]
            ixz_ori = grid[start_idx_tt:end_idx_tt, 2]

            warped_x[current_batch, :, ixx, ixy, ixz] = x[current_batch, :, ixx_ori, ixy_ori, ixz_ori]
            
        # for i in range(transformed_grid.shape[0]):  
        #     current_batch = batch_ix[i]
        #     ixx = transformed_grid[i, 0]
        #     ixy = transformed_grid[i, 1]
        #     ixz = transformed_grid[i, 2]

        #     ixx_ori = grid[i, 0]
        #     ixy_ori = grid[i, 1]
        #     ixz_ori = grid[i, 2]

        #     warped_x[current_batch, :, ixx, ixy, ixz] = x[current_batch, :, ixx_ori, ixy_ori, ixz_ori]

        return warped_x