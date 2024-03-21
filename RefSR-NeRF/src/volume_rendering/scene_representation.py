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
"""NeRF explicit 3D model scene representation."""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import ms_class


class NERFLayer(nn.Cell):
    """Fully connected NeRF layer for 3d model explicit representation."""

    def __init__(self, depth=8, width=256,
                 input_ch=3, input_ch_views=3, output_ch=4,
                 skips=(4,), use_view_dirs=False):
        super().__init__(auto_prefix=False)
        self.depth = depth
        self.input_ch = input_ch
        # self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_view_dirs = use_view_dirs

        # Ops.
        self.concat = ms.ops.Concat(axis=-1)

        # Layers.
        self.relu = nn.ReLU()
        pts_linears = [nn.Dense(input_ch, width, activation='relu')]
        for i in range(self.depth - 1):
            if i in self.skips:
                pts_linears.append(nn.Dense(width + input_ch,
                                            width,
                                            activation='relu'))
            else:
                pts_linears.append(nn.Dense(width, width, activation='relu'))
        self.pts_linears = nn.CellList(pts_linears)

        self.views_linears = nn.CellList([
            nn.Dense(input_ch_views + width, width // 2, activation='relu')
        ])  # Note: for torch weights loading.

        if self.use_view_dirs:
            self.alpha_linear = nn.Dense(width, 1)
            self.feature_linear = nn.Dense(width, width)
            self.rgb_linear = nn.Dense(width // 2, 3)
        else:
            self.output_linear = nn.Dense(width, output_ch)

    def construct(self, x):
        output = x[..., :self.input_ch]
        for i in range(self.depth):
            output = self.pts_linears[i](output)
            if i in self.skips:
                output = self.concat((x[..., :self.input_ch], output))

        if self.use_view_dirs:
            alpha = self.alpha_linear(output)
            feature = self.feature_linear(output)
            output = self.concat((feature, x[..., self.input_ch:]))

            output = self.views_linears[0](output)

            rgb = self.rgb_linear(output)
            output = self.concat((rgb, alpha))
        else:
            output = self.output_linear(output)
        return output


@ms_class
class Embedder:

    def __init__(self, multires: int,
                 input_dims: int = 3,
                 log_sampling: bool = True,
                 dtype=ms.float32):
        max_freq = multires - 1
        num_freqs = multires

        if log_sampling:
            freq_bands = ops.LinSpace()(ms.Tensor(0, dtype=dtype),
                                        ms.Tensor(max_freq, dtype=dtype),
                                        num_freqs)
            self.freq_bands = ops.Pow()(ms.Tensor(2.0, dtype=dtype),
                                        freq_bands)
        else:
            self.freq_bands = ops.LinSpace()(
                ms.Tensor(1, dtype=dtype),
                ms.Tensor(2 ** max_freq, dtype=dtype),
                num_freqs)
        self.output_dim = input_dims * (self.freq_bands.shape[0] * 2 + 1)

        self.concat = ops.Concat(axis=-1)

    def embed(self, inputs):
        v = inputs
        for f in self.freq_bands:
            v = self.concat((v, ops.Sin()(inputs * f)))
            v = self.concat((v, ops.Cos()(inputs * f)))
        return v


class PosAndDirSceneRepr(nn.Cell):

    def __init__(self,
                 multires_positions: int,
                 multires_directions: int,
                 nerf_depth: int,
                 nerf_width: int,
                 nerf_chunk: int,
                 output_channels: int):
        super().__init__()
        self.nerf_chunk = nerf_chunk
        self.pos_embedder = Embedder(multires_positions)
        self.dir_embedder = Embedder(multires_directions)
        self.nerf_layer = NERFLayer(
            depth=nerf_depth,
            width=nerf_width,
            input_ch=self.pos_embedder.output_dim,
            input_ch_views=self.dir_embedder.output_dim,
            output_ch=output_channels,
            use_view_dirs=True
        )
        self.nerf_concat = ops.Concat(axis=0)

    def construct(self, positions, directions):
        # Positions processing.
        pos_shape = positions.shape
        positions = ops.Reshape()(positions, (-1, pos_shape[-1]))
        positions = self.pos_embedder.embed(positions)

        # Direction embedding.
        tiled_dir = ops.Tile()(directions[:, None, :],
                               (1, pos_shape[1], 1))
        directions = ops.Reshape()(tiled_dir, (-1, directions.shape[-1]))
        directions = self.dir_embedder.embed(directions)

        # Concat embeddings and apply model.
        embedded = ops.Concat(axis=-1)((positions, directions))

        # Raise MS Error:
        res = self.nerf_layer(embedded[:self.nerf_chunk])
        if self.nerf_chunk < embedded.shape[0]:
            for i in range(self.nerf_chunk, embedded.shape[0], self.nerf_chunk):
                res = self.nerf_concat(
                    (res, self.nerf_layer(embedded[i: i + self.nerf_chunk]))
                )
        # res = self.nerf_layer(embedded)
        output_shape = tuple(list(pos_shape[:-1]) + [4])
        nerf_rerp = ops.Reshape()(res, output_shape)
        return nerf_rerp
