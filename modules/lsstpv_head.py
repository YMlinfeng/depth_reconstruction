import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from modules.occformer.occnet import OccupancyEncoder
from modules.image2bev.ViewTransformerLSSVoxel import (
    ViewTransformerLiftSplatShootVoxel,
)
from modules.render import RaySampler
from modules.render.render_utils import nerf_models
from modules.render.render_utils.rays import RayBundle


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class LSSTPVHead(nn.Module):
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
        render_scale=2,
        ray_sample_mode='fixed',
        num_imgs=3,
    ):
        super(LSSTPVHead, self).__init__()

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

        # 定义sampler
        self.ray_sampler = RaySampler(
                ray_sample_mode=ray_sample_mode,
                ray_number=[pad_size[0] // render_scale, pad_size[1] // render_scale],
                ray_img_size=pad_size,
                )
        
        # 定义render model
        self.render_model = getattr(nerf_models, "NeuSModel")(
            pc_range=np.array(point_cloud_range, dtype=np.float32),
            voxel_size=np.array([voxel_size, voxel_size, voxel_size], dtype=np.float32),
            voxel_shape=np.array([volume_h[0], volume_w[0], volume_z[0]], dtype=np.float32),
            norm_scene=True,
            field_cfg=dict(
                type="SDFField",
                beta_init=0.3,
                ),
            collider_cfg=dict(
                type="AABBBoxCollider", 
                near_plane=0.5,
                ),
            sampler_cfg=dict(
                type="NeuSSampler",
                initial_sampler="UniformSampler",
                num_samples=72,
                num_samples_importance=24,
                num_upsample_steps=1,
                train_stratified=True,
                single_jitter=True,
                ),
            loss_cfg=None,
            )
        
        self.num_imgs = num_imgs
        self._init_layers()

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

        # obtain lidar2cam and lidar2img
        sdf_preds = sdf_preds[-1:]
        lidar2img, lidar2cam = [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
        lidar2img = torch.from_numpy(np.asarray(lidar2img)).to(image_feats)
        lidar2cam = torch.from_numpy(np.asarray(lidar2cam)).to(image_feats)
        
        # render depth
        rays = self.sample_rays(image_feats, lidar2cam, lidar2img)
        render_depths = []
        for i in range(len(sdf_preds)):
            render_depth = []
            for bs_idx in range(len(lidar2cam)):
                i_ray_o, i_ray_d, i_ray_depth, scaled_points = (    # i_ray_o: n * num_points, 3 (6 * 512, 3)
                    rays[bs_idx]["ray_o"],
                    rays[bs_idx]["ray_d"],
                    rays[bs_idx].get("depth", None),
                    rays[bs_idx].get("scaled_points", None)
                )
                ray_bundle = RayBundle(
                        origins=i_ray_o, directions=i_ray_d, depths=None # i_ray_depth
                    )

                preds_dict = self.render_model(
                    ray_bundle, sdf_preds[i][bs_idx].permute(0, 3, 2, 1).contiguous(), points=scaled_points)

                # 点到相机平面的depth
                render_depth.append(self.depth2plane(i_ray_o, i_ray_d, preds_dict['depth'], lidar2img[bs_idx]))

            render_depths.append(torch.stack(render_depth, 0))

        # return *sdf_preds, depth
        return sdf_preds[-1], outputs[-1], render_depths[-1]


    def depth2plane(self, ray_o, ray_d, depth, lidar2img):
        lidar_points = ray_d * depth / self.render_model.scale_factor
        lidar_points = lidar_points + ray_o / self.render_model.scale_factor
        lidar_points = lidar_points.view(self.num_imgs, -1, 3)
        lidar_points = torch.cat([lidar_points, torch.ones(lidar_points[..., :1].shape).to(lidar_points)], -1)
        img_points = torch.matmul(lidar2img.unsqueeze(1), lidar_points.unsqueeze(-1))
        return torch.clamp(img_points.squeeze().view(-1, 4)[:, 2:3], min=1e-5)
    
    
    def sample_rays(self, feats, lidar2cam, lidar2img):
        batch_ret = []
        for bs_idx in range(len(feats)):
            i_lidar2img = lidar2img[bs_idx] # .flatten(0, 1)
            i_img2lidar = torch.inverse(
                i_lidar2img
            )  # TODO: Are img2lidar and img2cam consistent after image data augmentation?
            i_cam2lidar = torch.inverse(
                lidar2cam[bs_idx] # .flatten(0, 1)
            )

            i_sampled_ray_o, i_sampled_ray_d, i_sampled_depth, i_sampled_coords = (
                [],
                [],
                [],
                [],
            )

            for c_idx in range(self.num_imgs):

                j_sampled_all_pts, j_sampled_all_pts_cam = (
                    [],
                    [],
                )

                """ sample points """
                j_sampled_pts_cam = torch.round(self.ray_sampler())
                j_sampled_pts_cam = torch.cat([j_sampled_pts_cam, torch.ones_like(j_sampled_pts_cam[..., :1]), torch.ones_like(j_sampled_pts_cam[..., :1])], -1)

                j_sampled_pts = torch.matmul(
                    i_img2lidar[c_idx : c_idx + 1],
                    torch.cat(
                        [
                            j_sampled_pts_cam[..., :2]
                            * j_sampled_pts_cam[..., 2:3],
                            j_sampled_pts_cam[..., 2:],
                        ],
                        dim=-1,
                    ).unsqueeze(-1),
                ).squeeze(-1)[..., :3]
                j_sampled_all_pts.append(j_sampled_pts)
                j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :2])

                if len(j_sampled_all_pts) > 0:
                    """merge"""
                    j_sampled_all_pts = torch.cat(j_sampled_all_pts, dim=0)
                    j_sampled_all_pts_cam = torch.cat(j_sampled_all_pts_cam, dim=0)

                    unscaled_ray_o = i_cam2lidar[c_idx : c_idx + 1, :3, 3].repeat(
                        j_sampled_all_pts.shape[0], 1
                    )
                    i_sampled_ray_o.append(
                        unscaled_ray_o * self.render_model.scale_factor
                    )
                    i_sampled_ray_d.append(
                        F.normalize(j_sampled_all_pts - unscaled_ray_o, dim=-1)
                    )
                    i_sampled_coords.append(
                        j_sampled_all_pts_cam
                    )
                    sampled_depth = (
                        torch.norm(
                            j_sampled_all_pts - unscaled_ray_o, dim=-1, keepdim=True
                        )
                        * self.render_model.scale_factor
                    )
                    i_sampled_depth.append(sampled_depth)
            
            batch_ret.append(
                {
                "ray_o": torch.cat(i_sampled_ray_o, dim=0),
                "ray_d": torch.cat(i_sampled_ray_d, dim=0),
                "depth": torch.cat(i_sampled_depth, dim=0),
                "coords": torch.cat(i_sampled_coords, dim=0),
                }
                )

        return batch_ret