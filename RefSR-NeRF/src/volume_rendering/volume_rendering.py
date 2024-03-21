# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# This file or its part has been derived from the following repository
# and modified: https://github.com/yenchenlin/nerf-pytorch
# ============================================================================
"""Volume rendering pipeline."""

import mindspore as ms

from mindspore import nn
from mindspore import ops

from .coordinates_samplers import DepthCoordsBuilder, HierarchicalCoordsBuilder
from .scene_representation import PosAndDirSceneRepr


class NerfToOutput(nn.Cell):

    def __init__(self, white_bkgr: bool,
                 raw_noise_std: float = 0.0,
                 dtype=ms.float32):
        super().__init__()
        self.white_bkgr = white_bkgr
        self.raw_noise_std = raw_noise_std
        self.dtype = dtype

        # Ops.
        self.exp = ops.Exp()
        self.lpnorm = ops.LpNorm(axis=-1)
        self.relu = ms.nn.ReLU()
        self.sigmoid = ops.Sigmoid()

    def process(self, nerf_out, z_vals, rays_d):
        # Distance between nearest z coordinates.
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        # [N_rays, 1] add the last distance to keep dimensions.
        t = ops.Tile()(ms.Tensor([1e10], dtype=self.dtype),
                       dists[..., :1].shape)
        dists = ops.Concat(axis=-1)((dists, t))  # [N_rays, N_samples]
        # Scale along ray.
        dists = dists * self.lpnorm(rays_d[..., None, :])

        rgb = self.sigmoid(nerf_out[..., :3])
        noise = 0.0
        if self.raw_noise_std > 0.0 and self.training:
            noise = ms.numpy.rand(nerf_out[..., 3].shape) * self.raw_noise_std

        alpha = 1.0 - self.exp((-self.relu(nerf_out[..., 3] + noise) * dists))
        p_ones = ops.Ones()((alpha.shape[0], 1), self.dtype)
        p_not_alpha = 1.0 - alpha + 1e-10
        concatenated_p = ops.Concat(axis=-1)((p_ones, p_not_alpha))
        alpha_ray_sampling = ops.CumProd()(concatenated_p, -1)[:, :-1]
        weights = alpha * alpha_ray_sampling  # [N_rays, N_samples]

        rgb_map = ops.ReduceSum()(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = ops.ReduceSum()(weights * z_vals, -1)
        if self.white_bkgr:
            acc_map = ops.ReduceSum()(weights, -1)  # 0 for black, 1 for white
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        res = ops.Concat(axis=-1)((rgb_map, depth_map[..., None], weights))
        return res


class VolumeRendering(nn.Cell):

    def __init__(self,
                 multires_positions: int,
                 multires_directions: int,
                 depth_samples: int,
                 coarse_net_depth: int,
                 coarse_net_width: int,
                 hierarchical_samples: int,
                 fine_net_depth: int,
                 fine_net_width: int,
                 white_bkgr: bool = False,
                 dtype: ms.Type = ms.float32,
                 linear_disparity_sampling: bool = False,
                 perturbation: bool = False,
                 near: float = 0.0,
                 far: float = 1.0,
                 raw_noise_std: float = 0.0,
                 nerf_chunk: int = 1024 * 64):
        super().__init__()
        self.nerf_output_ch = 5
        self.nerf_coarse_model = PosAndDirSceneRepr(
            multires_positions=multires_positions,
            multires_directions=multires_directions,
            nerf_depth=coarse_net_depth,
            nerf_width=coarse_net_width,
            output_channels=self.nerf_output_ch,
            nerf_chunk=nerf_chunk
        )
        self.nerf_fine_model = PosAndDirSceneRepr(
            multires_positions=multires_positions,
            multires_directions=multires_directions,
            nerf_depth=fine_net_depth,
            nerf_width=fine_net_width,
            output_channels=self.nerf_output_ch,
            nerf_chunk=nerf_chunk
        )
        self.dtype = dtype
        self.near = near
        self.far = far
        self.depth_coords_builder = DepthCoordsBuilder(
            depth_samples, linear_disparity_sampling, perturbation, dtype)
        self.hierarchical_builder = HierarchicalCoordsBuilder(
            hierarchical_samples, perturbation, dtype)
        self.nerf2output = NerfToOutput(white_bkgr=white_bkgr,
                                        raw_noise_std=raw_noise_std,
                                        dtype=dtype)  # TODO: Check.

    def construct(self, rays):
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        raw_rays_d = rays[:, 6:]

        # Sample points.
        near = self.near * ops.OnesLike()(rays_d[..., :1]).astype(self.dtype)
        far = self.far * ops.OnesLike()(rays_d[..., :1]).astype(self.dtype)
        z_vals = self.depth_coords_builder.build_coords(near, far)
        positions = rays_d[..., None, :] * z_vals[..., :, None]
        positions = rays_o[..., None, :] + positions

        # Coarse nerf prediction.
        coarse_res = self.nerf2output.process(
            self.nerf_coarse_model(positions, raw_rays_d),
            z_vals,
            raw_rays_d)
        coarse_rgb, coarse_w = coarse_res[:, :3], coarse_res[:, 4:]

        # Add hierarchical sampler.
        z_values_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = self.hierarchical_builder.build_coords(
            z_values_mid, coarse_w[..., 1:-1])
        z_samples = ms.ops.stop_gradient(z_samples)
        z_values, _ = ops.Sort(axis=-1)(
            ops.Concat(axis=-1)((z_vals, z_samples)))
        positions = (rays_o[..., None, :] + rays_d[..., None, :] * z_values[..., :, None])
        # Fine nerf prediction.
        fine_res = self.nerf2output.process(
            self.nerf_fine_model(positions, raw_rays_d),
            z_values,
            rays_d)
        output = ops.Concat(axis=-1)((coarse_rgb, fine_res))
        return output
