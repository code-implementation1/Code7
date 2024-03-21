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
"""Convert weights from PyTorch NeRF repo to MindSpore."""

import argparse
import json
from pathlib import Path

import mindspore as ms
from mindspore import save_checkpoint
from mindspore.train.serialization import load_param_into_net

import torch

from src.volume_rendering.volume_rendering import VolumeRendering


def parse_args():
    """
    args function.
    Returns:
        args.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-config', type=Path,
                        help='Model config file.')
    parser.add_argument('--torch-weights', type=Path,
                        help='Torch checkpoint file.')
    parser.add_argument('--out-weights', type=Path,
                        help='Output mindspore checkpoint .ckpt file.')
    return parser.parse_args()


def load_torch_weights(model, torch_ckpt_path: Path, out_ckpt_path: Path):
    ckpt = torch.load(torch_ckpt_path)
    # load coarse params
    nerf_layer_coarse_params = {}
    for name, w in ckpt['network_fn_state_dict'].items():
        layer_name = 'nerf_coarse_model.nerf_layer.' + name
        nerf_layer_coarse_params[layer_name] = \
            ms.Parameter(w.cpu().numpy(), name=name)
    load_param_into_net(model.nerf_coarse_model.nerf_layer,
                        nerf_layer_coarse_params)

    # load fine params
    nerf_layer_fine_params = {}
    for name, w in ckpt['network_fine_state_dict'].items():
        layer_name = 'nerf_fine_model.nerf_layer.' + name
        nerf_layer_fine_params[layer_name] = \
            ms.Parameter(w.cpu().numpy(), name=name)
    load_param_into_net(model.nerf_fine_model.nerf_layer,
                        nerf_layer_fine_params)

    save_checkpoint(model, str(out_ckpt_path))


if __name__ == '__main__':
    args = parse_args()
    with open(args.model_config, 'r') as f:
        model_cfg = json.load(f)
    nerf = VolumeRendering(**model_cfg)
    load_torch_weights(nerf, args.torch_weights, args.out_weights)
