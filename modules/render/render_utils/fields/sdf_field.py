import torch
import torch.nn.functional as F
from torch import nn
from ... import grid_sample_3d
import numpy as np


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter(
            "beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False)
        )
        self.register_parameter(
            "beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, sdf, beta=None):
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS"""

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter(
            "variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(
            self.variance * 10.0
        )

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


class SDFField(nn.Module):
    def __init__(
        self,
        voxel_size,
        pc_range,
        voxel_shape,
        scale_factor,
        beta_init,
        **kwargs
    ):
        super().__init__()
        self.fp16_enabled = kwargs.get("fp16_enabled", False)
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.voxel_shape = voxel_shape
        self.beta_init = beta_init
        self.scale_factor = scale_factor

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.beta_init)

        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(init_val=self.beta_init)

        self._cos_anneal_ratio = 1.0

    def set_cos_anneal_ratio(self, anneal):
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def get_alpha(self, ray_samples, sdf, gradients):
        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self._cos_anneal_ratio)
            + F.relu(-true_cos) * self._cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

    def interpolate_feats(self, pts, feats_volume):
        pc_range = pts.new_tensor(self.pc_range)
        norm_coords = (pts / self.scale_factor - pc_range[:3]) / (
            pc_range[3:] - pc_range[:3]
        )
        assert (
            self.voxel_shape[0] == feats_volume.shape[3]
            and self.voxel_shape[1] == feats_volume.shape[2]
            and self.voxel_shape[2] == feats_volume.shape[1]
        )
        norm_coords = norm_coords * 2 - 1
        feats = (
            grid_sample_3d(feats_volume.unsqueeze(0), norm_coords[None, None, ...])
            .squeeze(0)
            .squeeze(1)
            .permute(1, 2, 0)
        )
        return feats
    
    def get_sdf(self, points, feature_volume):
        """predict the sdf value for ray samples"""
        sdf = self.interpolate_feats(points, feature_volume[:1])
        return sdf, sdf, sdf

    def get_density(self, ray_samples, feature_volume):
        """Computes and returns the densities."""
        points = ray_samples.frustums.get_start_positions()
        sdf, _, _ = self.get_sdf(points, feature_volume)
        density = self.laplace_density(sdf)
        return density

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = torch.sigmoid(-10.0 * sdf)
        return occupancy

    def forward(self, ray_samples, feature_volume, return_alphas=False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        outputs = {}

        points = ray_samples.frustums.get_start_positions()
        points.requires_grad_(True)
        with torch.enable_grad():
            sdf, geo_features, point_features = self.get_sdf(points, feature_volume)

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        directions = ray_samples.frustums.directions
        rgb = sdf.repeat(1, 1, 3)
        density = self.laplace_density(sdf)

        outputs.update(
            {
                "rgb": rgb,
                "density": density,
                "sdf": sdf,
                "gradients": gradients,
            }
        )

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({"alphas": alphas})

        return outputs