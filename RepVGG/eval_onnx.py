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
"""Run evaluation for ONNX model."""
import argparse
from pathlib import Path

import onnxruntime as ort
from mindspore import nn
from mindspore import dataset as ds

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
    parser.add_argument('dataset', type=Path,
                        help='Path to dataset for prediction.')
    parser.add_argument('-c', '--onnx_path', type=Path,
                        help='Path ONNX saved model.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size.')
    parser.add_argument(
        '-m', '--device_target', choices=['CPU', 'GPU'], default='CPU',
        help='Target computation platform.'
    )

    parser.add_argument(
        '--prefetch', default=16, type=int, help='Prefetch images.'
    )

    return parser.parse_args()


def create_session(onnx_path, device_target):
    """Create ONNX inference session."""
    if device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {device_target}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_eval(args):
    """Run evaluation for ONNX model."""
    session, input_name = create_session(
        str(args.onnx_path), args.device_target
    )
    args.batch_size = 1
    ds.config.set_prefetch_size(args.prefetch)
    dataset = ds.ImageFolderDataset(str(args.dataset.absolute()),
                                    shuffle=False, decode=True)
    preprocess = get_transforms(image_size=args.image_size, training=False)
    dataset = dataset.map(input_columns=['image'], operations=preprocess)
    dataset = dataset.batch(args.batch_size)
    iterator = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    metrics = {
        'Top1-Acc': nn.Top1CategoricalAccuracy(),
        'Top5-Acc': nn.Top5CategoricalAccuracy(),
    }
    for batch in iterator:
        y_pred = session.run(None, {input_name: batch['image']})[0]
        for metric in metrics.values():
            metric.update(y_pred, batch['label'])
    return {name: metric.eval() for name, metric in metrics.items()}


def main():
    """Entry point."""
    args = parse_args()
    results = run_eval(args)
    print(f'=> eval results:{results}')
    print('=> eval success')


if __name__ == '__main__':
    main()
