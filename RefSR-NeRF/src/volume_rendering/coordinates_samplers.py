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
"""MindSpore based coordinates sampling: depth and hierarchical samplers."""

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops


class DepthCoordsBuilder(nn.Cell):

    def __init__(self,
                 depth_samples: int,
                 linear_disparity_sampling: bool,
                 perturbation: bool,
                 dtype=ms.float32):
        super().__init__()
        self.depth_samples = depth_samples
        self.linear_disparity_sampling = linear_disparity_sampling
        self.perturbation = perturbation
        self.dtype = dtype

        zero_tensor = ms.Tensor(0.0, dtype=self.dtype)
        one_tensor = ms.Tensor(1.0, dtype=self.dtype)
        self.t_vals = ops.LinSpace()(zero_tensor,
                                     one_tensor,
                                     self.depth_samples)

    def build_coords(self, near, far):
        if not self.linear_disparity_sampling:
            z_vals = near * (1.0 - self.t_vals) + far * self.t_vals
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - self.t_vals) +
                            1.0 / far * self.t_vals)
        if self.perturbation and self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = ms.ops.Concat(axis=-1)((mids, z_vals[..., -1:]))
            lower = ms.ops.Concat(axis=-1)((z_vals[..., :1], mids))
            t_rand = ms.numpy.rand(z_vals.shape, dtype=self.dtype)
            z_vals = lower + (upper - lower) * t_rand
        return z_vals


class HierarchicalCoordsBuilder(nn.Cell):

    def __init__(self, hierarchical_samples: int,
                 perturbation: bool,
                 dtype=ms.float32):
        super().__init__()
        self.hierarchical_samples = hierarchical_samples
        self.perturbation = perturbation
        self.dtype = dtype

        self.linspace = ops.LinSpace()(ms.Tensor(0, dtype=self.dtype),
                                       ms.Tensor(1, dtype=self.dtype),
                                       self.hierarchical_samples)

        self.concat = ops.Concat(axis=-1)
        self.cumsum = ops.CumSum()
        self.gatherd = ops.GatherD()
        self.max = ops.Maximum()
        self.min = ops.Minimum()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.stack = ops.Stack(axis=-1)
        self.tile = ops.Tile()

    def build_coords(self, bins, weights):
        weights = weights + 1e-5
        pdf = weights / self.reduce_sum(weights, -1)
        cdf = ops.CumSum()(pdf, -1)
        zeros_like = ops.ZerosLike()(cdf[..., :1])
        cdf = self.concat((zeros_like, cdf))  # (batch, len(bins))

        if self.perturbation and self.training:
            u = ms.numpy.rand(list(cdf.shape[:-1]) +
                              [self.hierarchical_samples])
        else:
            u = self.tile(self.linspace, tuple(list(cdf.shape[:-1]) + [1]))
        inds = ops.SearchSorted(right=True, dtype=ms.int32)(cdf, u)

        below = self.max(ops.ZerosLike()(inds - 1), inds - 1)
        min_avlbl = (cdf.shape[-1] - 1) * ops.OnesLike()(inds)
        above = self.min(min_avlbl, inds)

        inds_g = self.stack((below, above))  # (batch, fine_samples, 2)

        tiled_cdf = self.tile(cdf[:, None, :],
                              (1, self.hierarchical_samples, 1))
        cdf_g = self.gatherd(tiled_cdf, 2, inds_g)
        tiled_bins = self.tile(bins[:, None, :],
                               (1, self.hierarchical_samples, 1))
        bins_g = self.gatherd(tiled_bins, 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = ms.numpy.where(denom < 1e-5, ops.OnesLike()(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        return samples
