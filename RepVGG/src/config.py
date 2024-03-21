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
"""Global configurations and command line flags parsing for RepVGG.

Value priority for parameter:
1) Command line value
2) Yaml value
3) Command line parser default value.
"""
import argparse
import os
import sys

from pathlib import Path

import yaml

args = None
path_args_list = [
    'config', 'train_url', 'data_url', 'pretrained', 'onnx_path', 'continues',
    'export_path', 'pred_output', 'dataset_path'
]


def _parse_path(parser):
    # main arguments
    parser.add_argument(
        '--config', default='./src/configs/RepVGG-A0.yaml',
        type=str, required=True,
        help='Path to config file to use (see configs dir).'
    )
    parser.add_argument(
        '--train_url', default='.', type=str,
        help='Location of training outputs.'
    )
    parser.add_argument(
        '--data_url', type=str, help='Path to data.'
    )


def _parse_resume_training(parser):
    # continue training or model inference
    parser.add_argument(
        '--pretrained', dest='pretrained', default=None, type=str,
        help='Path to pre-trained model checkpoint (if used).'
    )
    parser.add_argument(
        '--continues', type=str, default=None,
        help='Save checkpoints to this folder. Is used to set folder where '
             'checkpoints and logs will be saved. If not set, folder will be '
             'created automatically. Set folder to continue training if it '
             'was stopped unexpectedly.'
    )
    parser.add_argument(
        '--exclude_epoch_state', type=int, default=0, choices=[0, 1],
        help='Exclude epoch state and learning rate.'
    )
    parser.add_argument(
        '--onnx_path', type=str, default=None, help='ONNX file path.'
    )


def _parse_arch_params(parser):
    # Model architecture
    parser.add_argument(
        '-a', '--arch', metavar='ARCH', default='RepVGG-B0',
        help='Model architecture name.'
    )
    parser.add_argument(
        '--num_classes', default=1000, type=int, help='Number of classes.'
    )
    parser.add_argument(
        '--image_size', default=224, type=int,
        help='Image size (size*size).'
    )
    parser.add_argument(
        '--deploy', type=int, default=0, choices=[0, 1],
        help='Whether create model in deploy mode.'
    )


def _parse_training_time_params(parser):
    # Training duration settings
    parser.add_argument(
        '--epochs', default=120, type=int, metavar='N',
        help='Number of total epochs to run.'
    )
    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N',
        help='Manual start epoch number (useful on restarts)'
    )
    parser.add_argument(
        '--start_step', default=0, type=int, metavar='N',
        help='Manual step of start epoch number.'
    )
    parser.add_argument(
        '--batch_size', default=32, type=int, metavar='N',
        help='Mini-batch size (default: 32), this is the total batch size of '
             'all Devices on the current node when using Data Parallel or '
             'Distributed Data Parallel'
    )


def _parse_checkpoint_params(parser):
    # Checkpoints setting
    parser.add_argument(
        '--keep_checkpoint_max', default=10, type=int,
        help='Max number of saved checkpoints (n last checkpoints. interval '
             'between checkpoint is set due "--save_every").'
    )
    parser.add_argument(
        '--save_every', default=5000, type=int,
        help='Save checkpoint every m training steps. Max number of steps is '
             'defined due "--keep_checkpoint_max" argument.'
    )
    parser.add_argument(
        '--keep_best_checkpoints_max', default=5, type=int,
        help='Max number of saved checkpoints (n best checkpoints).'
    )


def _parse_optimizer_params(parser):
    # Optimizer settings
    parser.add_argument(
        '--optimizer', default='momentum', choices=['momentum', 'adamw'],
        help='Used optimizer.'
    )
    parser.add_argument(
        '--warmup_length', default=0, type=int,
        help='Number of warmup iterations.'
    )
    parser.add_argument(
        '--warmup_lr', default=5e-7, type=float,
        help='Warm up learning rate.'
    )
    parser.add_argument(
        '--base_lr', default=0.1, type=float,
        help='Initial learning rate value in the start of training '
             '(after warmup).'
    )
    parser.add_argument(
        '--min_lr', default=1e-7, type=float,
        help='Final learning rate on the last step.'
    )
    parser.add_argument(
        '--momentum', default=0.9, type=float, metavar='M',
        help='Momentum coefficient for momentum optimizer.'
    )
    parser.add_argument(
        '--beta', default=[0.9, 0.999], type=float, nargs=2,
        help='Beta for AdamW optimizer.'
    )
    parser.add_argument(
        '--eps', default=1e-8, type=float, help='AdamW optimizer parameter.'
    )
    parser.add_argument(
        '--lr_scheduler', default='cosine_lr',
        choices=['constant_lr', 'cosine_lr', 'multistep_lr', 'exp_lr'],
        help='Scheduler for the learning rate.'
    )
    parser.add_argument(
        '--lr_adjust', default=30, type=float,
        help='Interval to drop learning rate (only for multistep_lr '
             'scheduler).'
    )
    parser.add_argument(
        '--lr_gamma', default=0.97, type=int,
        help='Multi-step multiplier (only for multistep_lr scheduler) '
             '(default: 0.97).'
    )
    parser.add_argument(
        '--wd', '--weight_decay', default=0.05, type=float, metavar='W',
        dest='weight_decay',
        help='Weight decay regularization coefficient (default: 0.05)',
    )


def _parse_ema_params(parser):
    # Train one step parameters (Exponential Moving Average and loss scale)
    parser.add_argument(
        '--loss_scale', default=1024, type=int, help='Loss scale.'
    )
    parser.add_argument(
        '--is_dynamic_loss_scale', default=1, type=int,
        help='Whether loss scale is dynamic.'
    )
    parser.add_argument(
        '--with_ema', type=int, default=0, choices=[0, 1],
        help='Training with EMA (Exponential Moving Average).'
    )
    parser.add_argument(
        '--ema_decay', default=0.9999, type=float,
        help='EMA (Exponential Moving Average) decay parameter.'
    )


def _parse_amp_params(parser):
    # Model optimization
    parser.add_argument(
        '--amp_level', default='O0', choices=['O0', 'O1', 'O2', 'O3'],
        help='AMP optimization level.'
    )


def _parse_hardware_params(parser):
    # Hardware setting
    parser.add_argument(
        '--device_id', default=0, type=int,
        help='Device id. Used when one there are not distributed training.'
    )
    parser.add_argument(
        '--device_num', default=1, type=int,
        help='Device num. Used for distributed training.'
    )
    parser.add_argument(
        '--device_target', default='GPU', choices=['GPU', 'CPU'],
        type=str, help='Platform (device) name.'
    )


def _parse_data_loading_settings(parser):
    # Data loading setting
    parser.add_argument(
        '-j', '--num_parallel_workers', default=4, type=int, metavar='N',
        help='Number of data loading workers (default: 4).'
    )
    parser.add_argument(
        '--prefetch', default=10, type=int, metavar='N',
        help='Number of prefetched batches per worker (default: 10).'
    )
    parser.add_argument(
        '--cache_session_id', type=int,
        help='Session id for cache admin server (Used to prefetch data).'
    )
    parser.add_argument(
        '--use_data_sink', type=int, default=0,
        help='Use data sink mode or no (it may involve to checkpoint '
             'saving).'
    )


def _parser_aug_params(parser):
    # Data augmentation settings
    parser.add_argument(
        '--aug_type', type=str, default='weak',
        choices=['none', 'weak', 'auto'],
        help='Type augmentation for training data set. \n"none" is simple '
             'preprocessing, \n"weak" is standard augmentation for '
             'lightweight architecture (random flip), \n"auto" is usage of '
             'auto augmentation.'
    )
    parser.add_argument(
        '--auto_augment', type=str, default='rand-m9-mstd0.5-inc1',
        help='Auto augmentation definition. If "--aug_type" is not "auto", '
             'this parameter is ignored.'
    )
    parser.add_argument(
        '--interpolation', default='bilinear', choices=['bilinear', 'bicubic'],
        help='Auto augmentation interpolation type.'
    )
    parser.add_argument(
        '--re_prob', default=0.0, type=float,
        help='Random erasing parameter. Probability of random erasing. '
             'Ignored if "--aug_type" is not "auto".'
    )
    parser.add_argument(
        '--re_count', default=1, type=int,
        help='Random erasing parameter. Number of erased fields. Ignored if '
             '--re_prob is 0.'
    )
    parser.add_argument(
        '--re_mode', choices=['pixel', 'rand', 'const'], default='pixel',
        help='Random erasing parameter. Filling type of erasing. Ignored if '
             '--re_prob is 0.'
    )

    parser.add_argument(
        '--mixup_prob', default=0.0, type=float,
        help='Probability of applying mixup or cutmix per batch or element.'
    )
    parser.add_argument(
        '--mix_up', default=0., type=float,
        help='Mixup alpha value, mixup is active if > 0'
    )
    parser.add_argument(
        '--switch_prob', default=0.0, type=float,
        help='Probability of switching to cutmix instead of mixup when both '
             'are active.'
    )
    parser.add_argument(
        '--cutmix', default=0.0, type=float,
        help='Cutmix alpha value, cutmix is active if > 0.'
    )
    parser.add_argument(
        '--mixup_mode', default='batch', choices=['batch', 'pair', 'elem'],
        help='How to apply mixup/cutmix params (per "batch", "pair" (pair of '
             'elements), "elem" (element).'
    )
    parser.add_argument(
        '--label_smoothing', type=float, default=0,
        help='Label smoothing to use, default 0.'
    )


def _parse_additional_params(parser):
    # Other
    parser.add_argument(
        '--file_format', type=str, choices=['AIR', 'MINDIR', 'ONNX'],
        default='MINDIR', help='File format exporting.'
    )
    parser.add_argument(
        '--export_path', type=str,
        default='model', help='Path to exported model file.'
    )
    parser.add_argument(
        '--pred_output', type=str,
        default='predictions.json', help='Path to predictions JSON file.'
    )
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Random seed for initializing training.'
    )
    parser.add_argument(
        '--brief', type=str, default=None, help='Training folder suffix.'
    )
    parser.add_argument(
        '--dataset_path', type=str,
        help='Path to folder with imagenet subset (for evaluation) or folder '
             'with images or single image (for inference).'
    )


def parse_arguments():
    """Parse arguments."""
    global args
    parser = argparse.ArgumentParser(description='MindSpore RepVGG Training.')

    _parse_path(parser)
    _parse_resume_training(parser)
    _parse_arch_params(parser)
    _parse_training_time_params(parser)
    _parse_checkpoint_params(parser)
    _parse_optimizer_params(parser)
    _parse_ema_params(parser)
    _parse_amp_params(parser)
    _parse_hardware_params(parser)
    _parse_data_loading_settings(parser)
    _parser_aug_params(parser)
    _parse_additional_params(parser)

    args = parser.parse_args()

    _get_config()


def _get_config():
    """Parse configuration from command line."""
    global args
    override_args = argv_to_vars(sys.argv)
    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)

    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    for key in loaded_yaml:
        if key in args.__dict__:
            args.__dict__[key] = loaded_yaml[key]
        else:
            raise ValueError(f'Config contains unsupported parameter {key}.')

    for name in path_args_list:
        if args.__dict__[name] is not None:
            args.__dict__[name] = Path(args.__dict__[name])

    os.environ["DEVICE_TARGET"] = args.device_target
    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(args.device_num)
    if "RANK_SIZE" not in os.environ.keys():
        os.environ["RANK_SIZE"] = str(args.device_num)


USABLE_TYPES = set([float, int])


def _trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    """Get variable name from arguments."""
    st = _trim_preceding_hyphens(st)
    st = st.replace('-', '_')

    return st.split('=')[0]


def argv_to_vars(argv):
    """Parse the command line."""
    var_names = []
    for arg in argv:
        if arg.startswith('-') and arg_to_varname(arg) != 'config':
            var_names.append(arg_to_varname(arg))
    return var_names


def run_args():
    """Run and get args."""
    global args
    if args is None:
        parse_arguments()

    return args
