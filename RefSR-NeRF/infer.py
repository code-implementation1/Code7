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
"""Inference based on the scene config, poses and trained model."""

import argparse
import json
from pathlib import Path

import imageio

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net

import numpy as np

from tqdm import tqdm

from src.tools.rays import get_rays, recalculate_rays_to_ndc
from src.volume_rendering import VolumeRendering


DEFAULT_MODEL_CONFIG = Path('src') / 'configs' / 'nerf_config.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--poses', type=Path, required=True,
                        help='Poses as camera to world matrix.')
    parser.add_argument('--scene-config', type=Path, required=True,
                        help='Scene config. '
                             'Contains at least: '
                             '* image size - width and heights;'
                             '* frame intrinsic matrix;'
                             '* near and far planes;'
                             '* white background or not;'
                             'Config can be obtained as the result of NeRF '
                             'training.')
    parser.add_argument('--model-config', type=Path,
                        default=DEFAULT_MODEL_CONFIG,
                        help='Volume rendering model config.')
    parser.add_argument('--model-ckpt', type=Path, required=True,
                        help='Model checkpoints.')
    parser.add_argument('--rays-batch', type=int, default=100,
                        help='Rays in batch size.')
    parser.add_argument('--out-path', type=Path, required=True,
                        help='Path for output data saving:'
                             'predicted image, configs, optionally video etc.')
    parser.add_argument('--export-video', action='store_true', default=False,
                        help='Build final video.')
    parser.add_argument('--mode', choices=['graph', 'pynative'],
                        default='graph',
                        help='Model representation mode. '
                             'Pynative for debugging.')
    return parser.parse_args()


def predict_cam2world_image(volume_renderer,
                            cam2world,
                            scene_config,
                            rays_batch):
    intrinsic = np.reshape(np.array(scene_config['intrinsic']), (3, 3))
    w, h = scene_config['width'], scene_config['height']

    # Build rays.
    rays_o, rays_d, rays_raw_d = get_rays(width=w,
                                          height=h,
                                          intrinsic=intrinsic,
                                          c2w=cam2world)
    if scene_config['is_ndc']:
        rays_o, rays_d = recalculate_rays_to_ndc(height=h,
                                                 width=w,
                                                 focal=scene_config['focal'],
                                                 near=1.0,
                                                 rays_o=rays_o,
                                                 rays_d=rays_d)
    rgb_coarse = []
    rgb_fine = []
    depth_fine = []
    weight_fine = []

    rays_o = ms.Tensor(rays_o, dtype=ms.float32)
    rays_d = ms.Tensor(rays_d, dtype=ms.float32)
    rays_raw_d = ms.Tensor(rays_raw_d, dtype=ms.float32)
    for i in tqdm(range(0, len(rays_o), rays_batch), leave=False):
        end = i + rays_batch
        r_o = rays_o[i: end]
        r_d = rays_d[i: end]
        raw_r_d = rays_raw_d[i: end]
        rays = ms.ops.Concat(axis=-1)((r_o, r_d, raw_r_d))
        res = volume_renderer(rays)
        # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
        #                           depth_map / torch.sum(weights, -1))
        res = res.asnumpy()
        rgb_coarse.append(res[:, :3])
        rgb_fine.append(res[:, 3:6])
        depth_fine.append(res[:, 6:7])
        weight_fine.append(res[:, 7:])
    rgb_coarse = np.reshape(np.concatenate(rgb_coarse, axis=0), (h, w, 3))
    rgb_fine = np.reshape(np.concatenate(rgb_fine, axis=0), (h, w, 3))
    depth_fine = np.reshape(np.concatenate(depth_fine, axis=0), (h, w, -1))
    weights = np.reshape(np.concatenate(weight_fine, axis=0), (h, w, -1))
    return rgb_coarse, rgb_fine, depth_fine, weights


def main():
    args = parse_args()

    mode = ms.GRAPH_MODE if args.mode == 'graph' else ms.PYNATIVE_MODE
    ms.set_context(mode=mode,
                   device_target='GPU')

    # Scene config.
    with open(args.scene_config, 'r') as f:
        scene_config = json.load(f)
    # Poses config.
    with open(args.poses, 'r') as f:
        cam2world = json.load(f)
        cam2world = np.reshape(np.array(cam2world), (-1, 4, 4))
    # Model config.
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)

    # Output directory.
    args.out_path.mkdir(parents=True, exist_ok=True)

    scene_config_out = args.out_path / 'scene_config.json'
    with open(scene_config_out, 'w') as f:
        json.dump(scene_config, f, indent=4)

    model_config_out = args.out_path / 'model_config.json'
    with open(model_config_out, 'w') as f:
        json.dump(model_config, f, indent=4)

    # Build model and load weights.
    volume_rendering = VolumeRendering(
        **model_config,
        linear_disparity_sampling=scene_config['linear_disparity_sampling'],
        white_bkgr=scene_config['white_background'],
        near=scene_config['near'],
        far=scene_config['far'],
        perturbation=False,
        raw_noise_std=0.0
    )
    ckpt = load_checkpoint(str(args.model_ckpt))
    load_param_into_net(volume_rendering, ckpt)

    fine_images = []
    disp_maps = []

    for pose_idx, pose in enumerate(tqdm(cam2world)):
        rgb_coarse, rgb_fine, depth_fine, weights = predict_cam2world_image(
            volume_rendering,
            pose,
            scene_config,
            args.rays_batch
        )
        # Save images and arrays.
        np.save(args.out_path / f'{pose_idx}_rgb_fine', rgb_fine)
        np.save(args.out_path / f'{pose_idx}_depth_fine', depth_fine)
        np.save(args.out_path / f'{pose_idx}_weight_fine', weights)

        img_coarse = np.uint8(np.clip(rgb_coarse, 0, 1) * 255)
        imageio.imwrite(args.out_path / f'{pose_idx}_image_coarse.png',
                        img_coarse)
        img_fine = np.uint8(np.clip(rgb_fine, 0, 1) * 255)
        imageio.imwrite(args.out_path / f'{pose_idx}_image_fine.png', img_fine)
        if args.export_video:
            fine_images.append(img_fine)
            disp_map = 1.0 / np.maximum(
                1e-10 * np.ones_like(depth_fine),
                depth_fine / (np.sum(weights, -1, keepdims=True) + 1e-6)
            )
            disp_map = np.uint8(
                np.clip(disp_map / np.max(disp_map), 0, 1) * 255
            )
            disp_maps.append(disp_map)

    if args.export_video:
        rgb_video = args.out_path / 'rgb_video.mp4'
        imageio.mimwrite(rgb_video, fine_images, fps=30, quality=8)

        disp_video = args.out_path / 'disp_video.mp4'
        imageio.mimwrite(disp_video, disp_maps, fps=30, quality=8)


if __name__ == '__main__':
    main()
