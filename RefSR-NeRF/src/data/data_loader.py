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

import numpy as np

from .ds_loaders import blender, llff


def load_data(ds_context, datadir, scale):
    K = None
    ds_type = ds_context.data_type
    if ds_type == 'llff':
        images, poses, bds, render_poses, i_test = llff.load_llff_data(
            datadir,
            ds_context.factor,
            recenter=True,
            bd_factor=.75,
            spherify=ds_context.spherify,
            scale=scale
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf,
              datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if ds_context.llffhold > 0:
            print('Auto LLFF holdout,', ds_context.llffhold)
            i_test = np.arange(images.shape[0])[::ds_context.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if not ds_context.is_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.0
            far = 1.0
        print('NEAR FAR', near, far)
        i_split = i_train, i_val, i_test

    elif ds_type == 'blender':
        images, poses, render_poses, hwf, i_split = \
            blender.load_blender_data(
                datadir,
                ds_context.half_res,
                ds_context.testskip
            )
        print('Loaded blender', images.shape, hwf, datadir)
        i_train, i_val, i_test = i_split

        near = 2.0
        far = 6.0

        if ds_context.white_background:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    else:
        raise IOError(f'Unknown dataset type: {ds_type}')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    print(f'{len(i_split[0])} train, '
          f'{len(i_split[1])} val, '
          f'{len(i_split[2])} test.')
    return images, poses, hwf, K, near, far, i_split
