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
"""Visualize augmentations."""
import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from src.dataset import get_transforms


def parse_args():
    """
    Create and parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-h', '--help', action='help',
                        default=argparse.SUPPRESS,
                        help='Show this help message and exit.')
    parser.add_argument('img', type=Path,
                        help='Path to dataset for prediction.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size.')
    parser.add_argument('--aug_type', type=str, default='weak',
                        choices=['none', 'weak', 'auto'],
                        help='Type augmentation for training data set. \n'
                             '"none" is simple preprocessing, \n'
                             '"weak" is standard augmentation for lightweight '
                             'architecture, \n'
                             '"auto" is usage of auto augmentation.')
    parser.add_argument('--auto_augment', type=str,
                        default='rand-m9-mstd0.5-inc1',
                        help='Auto augmentation definition.')
    parser.add_argument('--interpolation', default='bilinear',
                        choices=['bilinear', 'bicubic'])
    parser.add_argument(
        '--re_prob', default=0.0, type=float,
        help='Random erasing parameter. Probability of random erasing.'
    )
    parser.add_argument(
        '--re_count', default=1, type=int,
        help='Random erasing parameter. Number of erased fields.'
    )
    parser.add_argument(
        '--re_mode', choices=['pixel', 'rand', 'const'], default='pixel',
        help='Random erasing parameter. Filling type of erasing.'
    )

    parser.add_argument('--output', default=Path('image.png'), type=Path,
                        help='Path to augmentation result')

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    image = Image.open(str(args.img))

    transform = get_transforms(
        args.image_size, training=True,
        type=args.aug_type,
        interpolation=args.interpolation,
        auto_augment=args.auto_augment,
        re_prob=args.re_prob,
        re_mode=args.re_mode,
        re_count=args.re_count
    )
    y = image
    for t in transform:
        y = t(y)

    y -= y.min(axis=(1, 2), keepdims=True)
    y /= y.max(axis=(1, 2), keepdims=True)
    y[np.isnan(y)] = 0
    y = y.transpose([1, 2, 0])
    plt.imshow(y)
    plt.savefig(args.output)


if __name__ == '__main__':
    main()
