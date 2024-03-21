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
"""Evaluation using the test split of passed dataset."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, Tensor, ops
import numpy as np

import tqdm
from src.model.sr import SR
from src.data import load_data
from src.tools.rays import get_rays, recalculate_rays_to_ndc
from src.volume_rendering import VolumeRendering
from infer import predict_cam2world_image


DEFAULT_MODEL_CONFIG = Path('src') / 'configs' / 'nerf_config.json'


def parse_args():
    """
    args function.
    Returns:
        args.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', type=Path, required=True,
                        help='Dataset directory.')
    parser.add_argument('--data-config', type=Path, required=True,
                        help='Path to the dataset config.')
    parser.add_argument('--model-config', type=Path,
                        default=DEFAULT_MODEL_CONFIG,
                        help='Volume rendering model config.')
    parser.add_argument('--model-ckpt', type=Path, required=True,
                        help='Model checkpoints.')
    parser.add_argument('--sr-ckpt', type=Path, required=True,
                        help='Model checkpoints.')
    parser.add_argument('--rays-batch', type=int, default=400,
                        help='Rays in batch size.')
    parser.add_argument('--scale', type=int, default=4,
                        help='super resolution scale.')
    parser.add_argument('--out-path', type=Path, required=True,
                        help='Path for output data saving:'
                             'predicted image, configs, ')
    parser.add_argument('--mode', choices=['graph', 'pynative'],
                        default='graph',
                        help='Model representation mode. '
                             'Pynative for debugging.')
    return parser.parse_args()


class TestNerfDataset:
    """
    clss: dataset.
    """
    def __init__(self,
                 ds_path: Path,
                 ds_context,
                 rays_chunk_size: int,
                 scale: int):
        """
        The logic:
        index i means not an image index, but the chunk index.
        The size of dataset is the number of chunks to process the full images
        in the defined order.

        Idea: build rays and store them in the dataset.
        """
        images, poses, hwf, key, near, far, i_split = load_data(ds_context,
                                                              ds_path, scale)
        data_idx = i_split[2]
        self.images = images[data_idx]
        self.poses = poses[data_idx]
        self.height, self.width, self.focal = hwf
        self.intrinsic = key
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
        return rays, target, pose

    def __len__(self):
        return self.total_chunks


class DotDict(dict):
    """
    Idea: Dict
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():
    """
    Dict
    """
    args = parse_args()
    mode = ms.GRAPH_MODE if args.mode == 'graph' else ms.PYNATIVE_MODE
    ms.set_context(mode=mode,
                   device_target='GPU')

    # Load data.
    with open(args.data_config, 'r', encoding='utf-8') as f:
        data_cfg = json.load(f)
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_cfg = json.load(f)

    # Init dataset and model.
    test_ds = TestNerfDataset(args.data_path,
                              DotDict(data_cfg),
                              args.rays_batch,
                              args.scale)
    imgs = test_ds.images
    ref_img = Tensor(imgs[len(imgs)//2]).squeeze(0)
    ref_pose = test_ds.poses[len(imgs)//2]
    img_chunks = test_ds.image_chunks
    h, w = test_ds.height, test_ds.width
    volume_rendering = VolumeRendering(
        **model_cfg,
        linear_disparity_sampling=data_cfg['linear_disparity_sampling'],
        white_bkgr=data_cfg['white_background'],
        near=test_ds.near,
        far=test_ds.far,
        perturbation=False,
        raw_noise_std=0.0)
    net_sr = SR(scale=args.scale, n_resblocks=args.n_resblocks, n_feats=args.n_feats)
    net_sr.set_train(False)
    ckpt = load_checkpoint(str(args.model_ckpt))
    load_param_into_net(volume_rendering, ckpt)
    load_param_into_net(net_sr, ckpt)

    # Calculate metric.
    test_ds = ms.dataset.GeneratorDataset(
        source=test_ds,
        column_names=['rays', 'target', 'pose'],
        column_types=[ms.float32, ms.float32, ms.float32],
        shuffle=False
    )
    _, ref_rgb_fine, _, _ = predict_cam2world_image(
            volume_rendering,
            ref_pose,
            data_cfg,
            args.rays_batch
        )
    ref_rgb_fine = Tensor(ref_rgb_fine).unsqueeze(0)
    args.out_path.mkdir(parents=True, exist_ok=True)

    metrics = {}
    rgb_pred = []
    rgb_target = []
    for i, (rays, target, pose) in tqdm.tqdm(enumerate(test_ds)):

        # render_pred = volume_rendering(rays)[:, 3:6]
        # rgb_pred.append(net_sr(render_pred).asnumpy())
        rgb_pred.append(volume_rendering(rays)[:, 3:6].asnumpy())
        rgb_target.append(target.asnumpy())

        # If we have finished the prediction for one image.
        if ((i + 1) % img_chunks) == 0:
            img_idx = i // img_chunks

            # Calculate psnr.
            pr_img = np.clip(np.concatenate(rgb_pred, axis=0), 0, 1)
            gt_img = np.concatenate(rgb_target, axis=0)

            # Fill the dict and update tmp list.
            rgb_pred = []
            rgb_target = []

            # Save predicted image.
            img_fine = np.reshape(np.uint8(np.clip(pr_img, 0, 1) * 255),
                                  (h, w, 3))
            img_fine_tensor = Tensor(img_fine).permute(1, 2, 0)
            img_fine_up = ops.interpolate(img_fine_tensor, scale_factor=args.scale, mode='bicubic').unsqueeze(0)
            cast_op = ops.Cast()
            data = ops.Concat(cast_op(img_fine_up, ref_img))
            img_sr = net_sr(img_fine_tensor.unsqueeze(0), data, ref_rgb_fine).squeeze(0).asnumpy()
            img_sr = np.reshape(np.uint8(np.clip(img_sr, 0, 1) * 255),
                                (h, w, 3))
            img_gt = np.reshape(np.uint8(np.clip(gt_img, 0, 1) * 255),
                                (h, w, 3))
            p = -10. * np.log10(np.mean(np.square(pr_img - gt_img)))
            metrics[img_idx] = {'pose': pose, 'psnr': p, 'img_idx': img_idx}
            plt.imsave(
                args.out_path / f'image_fine_{img_idx}_psnr_{p}.png',
                img_fine
            )
            plt.imsave(
                args.out_path / f'image_gt_{img_idx}_psnr_{p}.png',
                img_gt
            )

    # Write to csv.
    csv_metrics = args.out_path / 'metrics.csv'
    with open(csv_metrics, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['img_idx', 'pose', 'psnr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([v for v in metrics.values()])
        mean_metric_writer = csv.writer(csvfile)
        psnr = np.mean(np.array([v['psnr'] for v in metrics.values()]))
        mean_metric_writer.writerow([f'Mean_PSNR: {psnr}'])


if __name__ == '__main__':
    main()
