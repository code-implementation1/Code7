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
"""main model of wdsr"""

import mindspore
from mindspore import Tensor, Parameter, ops, nn
import mindspore.numpy as mnp
import numpy as np
from src.model.unet import UNetMedical
from src.model.deg import Degradation


def norm_except_dim(v, p, dim):
    '''
    weight_norm
    '''
    if dim == -1:
        return mnp.norm(v, p)
    if dim == 0:
        output_size = (v.shape[0],) + (1,) * (v.ndim - 1)
        return mnp.norm(v.view((v.shape[0], -1)), p, 1).view(output_size)
    if dim == (v.ndim - 1):
        output_size = (1,) * (v.ndim - 1) + (v.shape[v.ndim - 1])
        return mnp.norm(v.view((-1, v.shape[v.ndim - 1])), p, 0).view(output_size)
    return norm_except_dim(v.swapaxes(0, dim), p, dim).swapaxes(0, dim)

def _weight_norm(v, g, dim):
    return v * (g / norm_except_dim(v, 2, dim))


class MeanShift(mindspore.nn.Conv2d):
    """add or sub means of input data"""
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040),
                    rgb_std=(1.0, 1.0, 1.0),
                      sign=-1, dtype=mindspore.float32):
        std = mindspore.Tensor(rgb_std, dtype)
        weight = mindspore.Tensor(np.eye(3),
                               dtype).reshape(3, 3, 1, 1) / std.reshape(3, 1, 1, 1)
        bias = sign * rgb_range * mindspore.Tensor(rgb_mean, dtype) / std
        super().__init__(3, 3, kernel_size=1,
                                        has_bias=True, weight_init=weight,
                                        bias_init=bias)
        for p in self.get_parameters():
            p.requires_grad = False

class WeightNorm(nn.Cell):
    '''
    weight_norm
    '''
    def __init__(self, module, dim):
        super().__init__()
        if dim is None:
            dim = -1
        self.dim = dim
        self.module = module
        self.assign = ops.assign()
        # add g and v as new parameters and express w as g/||v|| * v
        self.param_g = Parameter(Tensor(norm_except_dim(self.module.weight, 2, dim)))
        self.param_v = Parameter(Tensor(self.module.weight.data))
        self.module.weight.set_data(_weight_norm(self.param_v, self.param_g, self.dim))
        self.use_weight_norm = True

    def construct(self, *inputs, **kwargs):
        '''
        forward
        '''
        if not self.use_weight_norm:
            return self.module(*inputs, **kwargs)
        self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
        return self.module(*inputs, **kwargs)


class Block(nn.Cell):
    """residual block"""
    def __init__(self, n_feats, kernel_size, block_feats, wn=None, act=nn.ReLU(True)):
        super().__init__()
        # act = nn.ReLU()
        # self.res_scale = 1
        body = []
        conv1 = nn.Conv2d(n_feats, block_feats,
                        kernel_size, padding=kernel_size//2,
                        pad_mode='pad', has_bias=True)
        conv2 = nn.Conv2d(block_feats, n_feats,
                          kernel_size, padding=kernel_size//2,
                          pad_mode='pad', has_bias=True)
        if wn is not None:
            conv1 = wn(conv1)
            conv2 = wn(conv2)
        body.append(conv1)
        body.append(act)
        body.append(conv2)
        self.body = nn.SequentialCell(body)

    def construct(self, x):
        '''
        forward
        '''
        res = self.body(x)
        return res


class Scale(nn.Cell):
    '''
    sclae
    '''
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = Parameter(Tensor([init_value]))

    def construct(self, x):
        '''
        forward
        '''
        res = x * self.scale
        return res

class PixelShuffle(nn.Cell):
    """perform pixel shuffle"""
    def __init__(self, upscale_factor):
        super().__init__()
        self.depth2space = mindspore.ops.DepthToSpace(upscale_factor)

    def construct(self, x):
        '''forward'''
        return self.depth2space(x)


class CALayer(nn.Cell):
    '''
    CAlayer
    '''
    def __init__(self, channel, reduction=1, use_hsigmoid=False):
        super().__init__()
        self.avg_pool = ops.AdaptiveAvgPool2d(1)
        if use_hsigmoid: # use hsigmoid instead of sigmoid
            self.conv_du = nn.SequentialCell(
                nn.Conv2d(channel, channel // reduction,
                          1, padding=0, pad_mode='pad',
                          has_bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1,
                          padding=0, pad_mode='pad',
                          has_bias=True),
                nn.HSigmoid())
        else:
            self.conv_du = nn.SequentialCell(
                nn.Conv2d(channel, channel // reduction, 1,
                          padding=0, pad_mode='pad',
                          has_bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1,
                          padding=0, pad_mode='pad',
                          has_bias=True),
                ops.sigmoid())

    def construct(self, x):
        '''forward'''
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SR(nn.Cell):
    """main structure of wdsr"""
    def __init__(self, scale=4, n_resblocks=8, n_feats=16):
        super().__init__()
        c_in = 3
        block_feats = 128
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)
        wn = WeightNorm(0)
        use_ca = True
        use_hsigmoid = False
        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(
                c_in, n_feats, 3,
                pad_mode='pad',
                padding=(3 // 2), has_bias=True)))

        # define body module
        body = []
        self.x_scale_list = nn.CellList()
        self.res_scale_list = nn.CellList()
        self.auxilary_scale_list = nn.CellList()
        for _ in range(n_resblocks):
            body.append(Block(n_feats, 3, block_feats, wn=wn))
            self.x_scale_list.append(Scale(1))
            self.res_scale_list.append(Scale(1))
            self.auxilary_scale_list.append(Scale(1))
        # define tail module
        tail = []
        out_feats = scale * scale * c_in
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2, pad_mode='pad', has_bias=True)))
        tail.append(PixelShuffle(scale))
        tail_s0 = []
        tail_s0.append(
            wn(nn.Conv2d(c_in, out_feats, 3, padding=3 // 2, pad_mode='pad', has_bias=True)))
        tail_s0.append(PixelShuffle(scale))
        skip = []
        skip.append(
            wn(nn.Conv2d(c_in, out_feats, 3, padding=3 // 2, pad_mode='pad', has_bias=True)))
        skip.append(PixelShuffle(scale))
        self.head = nn.SequentialCell(head)
        self.body = nn.SequentialCell(body)
        self.tail = nn.SequentialCell(tail)
        self.tail_s0 = nn.SequentialCell(tail_s0)
        self.skip = nn.SequentialCell(skip)

        self.fusion_conv_list = nn.CellList()
        for j in range(n_resblocks):
            if use_ca:
                tmp = nn.SequentialCell([nn.Conv2d(n_feats*(j+1),
                                                   n_feats, 1, padding=0,
                                                   pad_mode='pad',
                                                   has_bias=False),
                                                   CALayer(n_feats, 1,
                                                           use_hsigmoid=use_hsigmoid)])
            else:
                tmp = nn.Sequential(*[nn.Conv2d(n_feats*(j+1), n_feats, 1, padding=0, bias=False)])
            self.fusion_conv_list.append(tmp)
        self.refine = UNetMedical(3*c_in, 1)
        self.degradation = Degradation()

    def construct(self, x, data, ref_mlp):
        '''forward'''
        x = self.sub_mean(x) / 127.5
        s = self.skip(x)
        s0 = self.haed(x)
        s1 = s0
        state_list = []
        state_list.append(s0)
        for i, blo in enumerate(self.body):
            s0, s1 = s1, blo(s1)
            s1 = self.x_scale_list[i](s0) + self.res_scale_list[i](s1)
            s1 = s1 + self.auxilary_scale_list[i](
                self.fusion_conv_list[i](
                    ops.cat(state_list, axis=1)))
            state_list.append(s1)
        out = self.tail(s1)
        out = out + s
        out_de = self.degradation(data).squeeze(0)
        s2 = self.tail_s0(ref_mlp)
        s3 = ops.cat((out, s2, out_de), 1)
        res = self.refine(s3)
        x = self.add_mean(out * 127.5)
        out += res
        return out
