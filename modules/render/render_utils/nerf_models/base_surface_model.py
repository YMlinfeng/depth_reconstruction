import torch
from torch import nn
import torch.nn.functional as F
from ..renderers import RGBRenderer, DepthRenderer
from .. import scene_colliders
from .. import fields
from .. import ray_samplers
from abc import abstractmethod
import numpy as np


class SurfaceModel(nn.Module):
    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        field_cfg,
        collider_cfg,
        sampler_cfg,
        loss_cfg,
        norm_scene,
        **kwargs
    ):
        super().__init__()
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = True
        self.scale_factor = (
            1.0 / np.max(np.abs(pc_range)) if norm_scene else 1.0
        )  # select the max length to scale scenes
        field_type = field_cfg.pop("type")
        self.field = getattr(fields, field_type)(
            voxel_size=voxel_size,
            pc_range=pc_range,
            voxel_shape=voxel_shape,
            scale_factor=self.scale_factor,
            **field_cfg
        )
        collider_type = collider_cfg.pop("type")
        self.collider = getattr(scene_colliders, collider_type)(
            scene_box=pc_range, scale_factor=self.scale_factor, **collider_cfg
        )
        sampler_type = sampler_cfg.pop("type")
        self.sampler = getattr(ray_samplers, sampler_type)(**sampler_cfg)
        self.rgb_renderer = RGBRenderer()
        self.depth_renderer = DepthRenderer()
        self.loss_cfg = loss_cfg

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle, feature_volume):
        """_summary_

        Args:
            ray_bundle (RayBundle): _description_
            return_samples (bool, optional): _description_. Defaults to False.
        """

    def get_outputs(self, ray_bundle, feature_volume, **kwargs):
        samples_and_field_outputs = self.sample_and_forward_field(
            ray_bundle, feature_volume
        )

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        rgb = self.rgb_renderer(rgb=field_outputs["rgb"], weights=weights)
        depth = self.depth_renderer(ray_samples=ray_samples, weights=weights)

        outputs = {
            "rgb": rgb,
            "depth": depth,
            "weights": weights,
            "sdf": field_outputs["sdf"],
            "gradients": field_outputs["gradients"],
            "z_vals": ray_samples.frustums.starts,
        }

        """ add for visualization"""
        outputs.update({"sampled_points": samples_and_field_outputs["sampled_points"]})
        if samples_and_field_outputs.get("init_sampled_points", None) is not None:
            outputs.update(
                {
                    "init_sampled_points": samples_and_field_outputs[
                        "init_sampled_points"
                    ],
                    "init_weights": samples_and_field_outputs["init_weights"],
                    "new_sampled_points": samples_and_field_outputs[
                        "new_sampled_points"
                    ],
                }
            )

        # if self.training:
        #     if self.loss_cfg.get("sparse_points_sdf_supervised", False):
        #         sparse_points_sdf, _, _ = self.field.get_sdf(
        #             kwargs["points"].unsqueeze(0), feature_volume
        #         )
        #         outputs["sparse_points_sdf"] = sparse_points_sdf.squeeze(0)

        return outputs

    def forward(self, ray_bundle, feature_volume, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        ray_bundle = self.collider(ray_bundle)  # set near and far
        return self.get_outputs(ray_bundle, feature_volume, **kwargs)

    def loss(self, preds_dict, targets):
        return
