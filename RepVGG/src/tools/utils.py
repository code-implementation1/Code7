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
"""Additional tools."""
import os

from mindspore import context
from mindspore import nn
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.ema.train_one_step_with_ema import TrainOneStepWithEMA


def set_device(args):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get('DEVICE_NUM', 1))

    if device_target == 'Ascend':
        if device_num > 1:
            context.set_context(device_id=int(os.environ['DEVICE_ID']))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True
            )
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    elif device_target == 'GPU':
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    else:
        raise ValueError('Unsupported platform.')

    return rank


def pretrained(args, model, exclude_epoch_state=True):
    """"Load pretrained weights if args.pretrained is given"""
    if os.path.isfile(args.pretrained):
        print(
            '=> loading pretrained weights from "{}"'.format(args.pretrained)
        )
        param_dict = load_checkpoint(str(args.pretrained))
        for key, value in param_dict.copy().items():
            if 'linear' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)

        if exclude_epoch_state:
            state_params = [
                'scale_sense', 'global_step', 'momentum', 'learning_rate',
                'epoch_num', 'step_num'
            ]
            for state_param in state_params:
                if state_param in param_dict:
                    param_dict.pop(state_param)
        load_param_into_net(model, param_dict)
    else:
        print('=> no pretrained weights found at "{}"'.format(args.pretrained))


def get_train_one_step(args, net_with_loss, optimizer):
    """get_train_one_step cell"""
    if args.device_target == 'CPU':
        print('=> CPU platform doesn\'t support loss scale')
        return nn.TrainOneStepCell(net_with_loss, optimizer)
    if args.is_dynamic_loss_scale:
        print('=> Using DynamicLossScaleUpdateCell')
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(
            loss_scale_value=2 ** 24, scale_factor=2, scale_window=2000
        )
    else:
        print(
            f'=> Using FixedLossScaleUpdateCell, '
            f'loss_scale_value:{args.loss_scale}'
        )
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(
            loss_scale_value=args.loss_scale
        )
    if args.with_ema:
        net_with_loss = TrainOneStepWithEMA(
            net_with_loss, optimizer, scale_sense=scale_sense,
            with_ema=args.with_ema, ema_decay=args.ema_decay
        )
    else:
        net_with_loss = nn.TrainOneStepWithLossScaleCell(
            net_with_loss, optimizer, scale_sense=scale_sense
        )
    return net_with_loss
