#!/usr/bin/env python
# -*- coding: UTF-8 -*
# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Run prediction on folder or single image, output results and save them to
JSON file.
"""
import json
import os

from functools import reduce
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from PIL import Image
from src.dataset import get_transforms
from src.tools.amp import cast_amp
from src.repvgg import get_model, switch_net_to_deploy


class NetWithSoftmax(nn.Cell):
    """
    Network with softmax at the end.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.softmax = nn.Softmax()

    def construct(self, x):
        res = self.softmax(self.net(x))
        return res


def data_loader(path: Path, image_size: int):
    """Load image or images from folder in generator."""
    preprocess = get_transforms(image_size=image_size,
                                training=False)

    def apply(img):
        for p in preprocess:
            img = p(img)
        return img
    extensions = ('.png', '.jpg', '.jpeg')
    if path.is_dir():
        print('=' * 5, ' Load directory ', '=' * 5)
        for item in path.iterdir():
            if item.is_dir():
                continue
            if item.suffix.lower() not in extensions:
                continue
            image = Image.open(str(item))
            image = image.convert('RGB')
            image = apply(image)
            yield str(item), ms.Tensor(image[None])
    else:
        print('=' * 5, ' Load single image ', '=' * 5)
        assert path.suffix.lower() in extensions

        image = Image.open(str(path))
        image = image.convert('RGB')
        image = apply(image)
        yield str(path), ms.Tensor(image[None])


def main():
    """Entry point."""
    from src.config import run_args
    args = run_args()

    os.environ["RANK_SIZE"] = '0'

    loader = data_loader(args.dataset_path, args.image_size)

    d = {}

    if args.pretrained is None or args.pretrained.suffix == '.ckpt':
        print('=== Use checkpoint ===')
        net = get_model(args)
        cast_amp(net, args)
        if args.pretrained:
            ms.load_checkpoint(str(args.pretrained.absolute()), net=net)
        print(
            'Number of parameters (before deploy):',
            sum(
                reduce(lambda x, y: x * y, params.shape)
                for params in net.trainable_params()
            )
        )
        switch_net_to_deploy(net)
        print(
            'Number of parameters (after deploy):',
            sum(
                reduce(lambda x, y: x * y, params.shape)
                for params in net.trainable_params()
            )
        )
        cast_amp(net, args)
        net = NetWithSoftmax(net)
    elif args.pretrained.suffix == '.mindir':
        print('=== Use MINDIR model ===')
        graph = ms.load(str(args.pretrained))
        net = nn.GraphCell(graph)
    else:
        raise ValueError(
            f'Unsupported checkpoint file format for "{args.pretrained}".'
        )

    context.set_context(
        mode=context.GRAPH_MODE, device_target=args.device_target,
    )

    argmax = ms.ops.Argmax(output_type=ms.int32)
    for (name, img) in loader:
        res = argmax(net(img)[0])
        print(name, f'(class: {res})')
        d[name] = int(res)

    with args.pred_output.open(mode='w') as f:
        json.dump(d, f, indent=1)


if __name__ == '__main__':
    main()
