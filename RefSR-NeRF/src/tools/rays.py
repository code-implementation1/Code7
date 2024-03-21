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
"""Rays sampling utilities."""

import numpy as np


def get_rays(height: int,
             width: int,
             intrinsic: np.ndarray,
             c2w: np.ndarray):
    i, j = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32),
                       indexing='xy')
    dirs = np.stack([(i - intrinsic[0][2]) / intrinsic[0][0],
                     -(j - intrinsic[1][2]) / intrinsic[1][1],
                     -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame.
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    rays_raw_d = rays_d / np.linalg.norm(rays_d,
                                         axis=-1,
                                         keepdims=True,
                                         ord=2)
    rays_raw_d = np.reshape(rays_raw_d, (-1, 3))
    rays_o = np.reshape(rays_o, (-1, 3))
    rays_d = np.reshape(rays_d, (-1, 3))
    return rays_o, rays_d, rays_raw_d


def recalculate_rays_to_ndc(height: int,
                            width: int,
                            focal: float,
                            near: float,
                            rays_o: np.ndarray,
                            rays_d: np.ndarray):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection.
    o0 = -1.0 / (width / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (height / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    x_div_z_d = rays_d[..., 0] / rays_d[..., 2]
    x_div_z_o = rays_o[..., 0] / rays_o[..., 2]
    d0 = -1.0 / (width / (2.0 * focal)) * (x_div_z_d - x_div_z_o)

    y_div_z_d = rays_d[..., 1] / rays_d[..., 2]
    y_div_z_o = rays_o[..., 1] / rays_o[..., 2]
    d1 = -1.0 / (height / (2.0 * focal)) * (y_div_z_d - y_div_z_o)

    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = np.stack([o0, o1, o2], -1)
    rays_d = np.stack([d0, d1, d2], -1)
    return rays_o, rays_d
