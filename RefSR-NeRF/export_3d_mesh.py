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
# and modified: https://github.com/bmild/nerf
# ============================================================================
"""Export 3d mesh: useful for blender dataset with synthetic scenes."""

import argparse
import json
from pathlib import Path

import mcubes
import trimesh

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net

from src.volume_rendering import VolumeRendering


DEFAULT_MODEL_CONFIG = Path('src') / 'configs' / 'nerf_config.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-config', type=Path,
                        default=DEFAULT_MODEL_CONFIG,
                        help='Path to the model config.')
    parser.add_argument('--model-ckpt', type=Path, required=True,
                        help='Model ckpt file.')
    parser.add_argument('--scene-config', type=Path, required=True,
                        help='Scene config file.')
    parser.add_argument('--points', type=int, default=256,
                        help='Number of points for each axis for mesh.')
    parser.add_argument('--batch', type=int, default=4096,
                        help='Points to pass to the fine model.')
    parser.add_argument('--out-stl', type=Path, required=True,
                        help='Directory to save training model meta info.')
    parser.add_argument('--mode', choices=['graph', 'pynative'],
                        default='graph',
                        help='Model representation mode. '
                             'Pynative for debugging.')
    return parser.parse_args()


def recalculate_to_ndc(pnts, focal, weights, heights, near):
    x_ratio = - focal / (weights / 2)
    y_ratio = - focal / (heights / 2)

    pnts[0] = pnts[0] / pnts[2] * x_ratio
    pnts[1] = pnts[1] / pnts[2] * y_ratio
    pnts[2] = 1 + 2 * near / pnts[2]
    return pnts


def main():
    args = parse_args()

    mode = ms.GRAPH_MODE if args.mode == 'graph' else ms.PYNATIVE_MODE
    ms.set_context(mode=mode)

    # Init volume rendering and load ckpt.
    with open(args.model_config, 'r') as f:
        model_cfg = json.load(f)
    with open(args.scene_config, 'r') as f:
        scene_cfg = json.load(f)

    volume_rendering = VolumeRendering(**model_cfg,
                                       near=scene_cfg['near'],
                                       far=scene_cfg['far'])
    ckpt = load_checkpoint(str(args.model_ckpt))
    load_param_into_net(volume_rendering, ckpt)

    # Build linspace and make the prediction.
    pts_min, pts_max = -1.2, 1.2
    t = np.linspace(pts_min, pts_max, args.points + 1)
    query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    flat_query_pts = ms.Tensor(query_pts.reshape([-1, 3]), dtype=ms.float32)
    viewdirs = ms.numpy.zeros_like(flat_query_pts, dtype=ms.float32)

    # Predict using the coarse scene representation.
    raw = []
    step = args.batch
    for i in tqdm(range(0, flat_query_pts.shape[0], step)):
        res = volume_rendering.nerf_fine_model(
            flat_query_pts[i: i + step, None, :],
            viewdirs[i: i + step]
        ).asnumpy()
        raw.append(res)
    raw = np.concatenate(raw, 0)
    raw = np.reshape(np.array(raw), list(query_pts.shape[:-1]) + [-1])
    sigma = np.maximum(raw[..., -1], 0.0)

    # Generate mesh.
    threshold = 50.0
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    mesh = trimesh.Trimesh(vertices / args.points - 0.5, triangles)
    mesh.export(file_obj=args.out_stl, file_type='stl')


if __name__ == '__main__':
    main()
