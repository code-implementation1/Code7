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

from pathlib import Path

import numpy as np

import mindspore as ms

from .data_loader import load_data

from ..tools.rays import get_rays, recalculate_rays_to_ndc


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TrainNerfDataset:

    def __init__(self,
                 ds_path: Path,
                 ds_context,
                 is_precrop: bool,
                 precrop_frac: float,
                 rays_chunk_size: int):
        images, poses, hwf, K, near, far, i_split = load_data(ds_context,
                                                              ds_path)
        train_idx = i_split[0]
        self.images = images[train_idx]
        self.poses = poses[train_idx]
        self.height, self.width, self.focal = hwf
        self.intrinsic = K
        self.far = far
        self.near = near

        # Need to convert to nds.
        self.is_ndc = ds_context['is_ndc']

        # Need to extract the random crop from image middle.
        self.is_precrop = is_precrop
        self.precrop_frac = precrop_frac

        # Rays chunk size to extract from each image.
        self.rays_chunk_size = rays_chunk_size

    def __getitem__(self, item):
        # Extract image and pose.
        target = np.reshape(self.images[item], (-1, 3)).astype(np.float32)
        pose = self.poses[item]

        # Generate rays.
        rays_o, rays_d, raw_rays_d = get_rays(self.height,
                                              self.width,
                                              self.intrinsic,
                                              pose)
        # Recalculate to nds if need.
        if self.is_ndc:
            rays_o, rays_d = recalculate_rays_to_ndc(self.height,
                                                     self.width,
                                                     self.focal,
                                                     1.0,
                                                     rays_o,
                                                     rays_d)
        # Build available rays coordinates.
        if self.is_precrop:
            # Choose image center and apply
            half_h = self.height // 2
            edge_h = int(half_h * self.precrop_frac)

            half_w = self.width // 2
            edge_w = int(half_w * self.precrop_frac)

            coords = np.stack(np.meshgrid(
                np.linspace(half_h - edge_h, half_h + edge_h - 1, 2 * edge_h),
                np.linspace(half_w - edge_w, half_w + edge_w - 1, 2 * edge_w)
            ), axis=-1)
        else:
            coords = np.stack(np.meshgrid(
                np.linspace(0, self.height - 1, self.height),
                np.linspace(0, self.width - 1, self.width)
            ), axis=-1)
        coords = np.reshape(coords, (-1, 2))

        # Choose random coordinates.is
        idx = np.random.choice(coords.shape[0],
                               size=[self.rays_chunk_size],
                               replace=False)
        chosen_coords = coords[idx]

        # Extract rays data and target by coordinates.
        flatten_idx = np.ravel_multi_index(
            chosen_coords.transpose().astype(np.int32),
            (self.height, self.width))
        rays_o = rays_o[flatten_idx]
        rays_d = rays_d[flatten_idx]
        raw_rays_d = raw_rays_d[flatten_idx]
        target = target[flatten_idx].astype(np.float32)
        rays = np.concatenate((rays_o, rays_d, raw_rays_d), axis=-1).astype(
            dtype=np.float32)
        return rays, target

    def __len__(self):
        return len(self.images)


class ValNerfDataset:

    def __init__(self,
                 ds_path: Path,
                 ds_context,
                 rays_chunk_size: int):
        """
        The logic:
        index i means not an image index, but the chunk index.
        The size of dataset is the number of chunks to process the full images
        in the defined order.

        Idea: build rays and store them in the dataset.

        :param ds_path:
        :param ds_type:
        :param ds_context:
        :param rays_chunk_size:
        """
        images, poses, hwf, K, near, far, i_split = load_data(ds_context,
                                                              ds_path)
        val_idx = i_split[1]  # TODO: Remove stub.
        self.images = images[val_idx]
        self.poses = poses[val_idx]
        self.height, self.width, self.focal = hwf
        self.intrinsic = K
        self.far = far
        self.near = near

        # Need to convert to ndc.
        self.is_ndc = ds_context['is_ndc']

        # Rays chunk size to extract from each image.
        img_size = self.width * self.height
        if img_size % rays_chunk_size != 0:
            raise IOError('Set the validation rays chunk size to be a divider '
                          'of validation image width * height.')
        self.rays_chunk_size = rays_chunk_size
        self.total_chunks = img_size * len(self.images) // self.rays_chunk_size
        self.image_chunks = img_size // self.rays_chunk_size

    def __getitem__(self, item):
        """Item means chunk i. Any image splits to chunk num rays."""
        start = self.rays_chunk_size * (item % self.image_chunks)
        end = start + self.rays_chunk_size

        # Get image split.
        img_id = item // self.image_chunks
        target = np.reshape(self.images[img_id], (-1, 3)
                            )[start: end].astype(np.float32)

        # Generate rays and choose split.
        pose = self.poses[img_id]
        rays_o, rays_d, raw_rays_d = get_rays(self.height,
                                              self.width,
                                              self.intrinsic,
                                              pose)
        # Recalculate to ndc if need.
        if self.is_ndc:
            rays_o, rays_d = recalculate_rays_to_ndc(self.height,
                                                     self.width,
                                                     self.focal,
                                                     1.0,
                                                     rays_o,
                                                     rays_d)
        rays_o = rays_o[start: end]
        rays_d = rays_d[start: end]
        raw_rays_d = raw_rays_d[start: end]
        rays = np.concatenate((rays_o, rays_d, raw_rays_d), axis=-1)
        return rays, target

    def __len__(self):
        return self.total_chunks


def create_datasets(ds_path: Path,
                    ds_context,
                    is_precrop: bool,
                    precrop_frac: float,
                    train_rays_chunk_size: int,
                    val_rays_chunk_size: int,
                    train_shuffle: bool = True,
                    num_parallel_workers: int = 1,
                    num_shards: int = 1,
                    shard_id: int = 0):
    ds_context = DotDict(ds_context)

    train_ds = TrainNerfDataset(ds_path,
                                ds_context,
                                is_precrop,
                                precrop_frac,
                                train_rays_chunk_size)
    scene_params = {
        'in_ndc': train_ds.is_ndc,
        'near': train_ds.near,
        'far': train_ds.far,
        'intrinsic': list(train_ds.intrinsic.reshape(-1)),
        'height': train_ds.height,
        'width': train_ds.width,
        'focal': float(train_ds.focal),
        'white_background': ds_context.white_background,
        'linear_disparity_sampling': ds_context.linear_disparity_sampling
    }
    train_ds = ms.dataset.GeneratorDataset(
        source=train_ds,
        column_names=['rays', 'target'],
        column_types=[ms.float32, ms.float32],
        shuffle=train_shuffle,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    val_ds = ValNerfDataset(ds_path,
                            ds_context,
                            val_rays_chunk_size)
    val_ds = ms.dataset.GeneratorDataset(
        source=val_ds,
        column_names=['rays', 'target'],
        column_types=[ms.float32, ms.float32],
        shuffle=False,
        num_parallel_workers=num_parallel_workers)
    return train_ds, val_ds, scene_params
