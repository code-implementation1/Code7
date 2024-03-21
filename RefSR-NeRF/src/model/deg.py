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
"""degredation branch"""


from mindspore import ops, nn
from mindediting.models.common.pixel_shuffle_pack import PixelShufflePack
from mindediting.models.common.resblock_with_input_conv import ResidualBlocksWithInputConv


def space2depth(block_size, x):
    '''
    method:space2depth
    '''
    b, t, c, h, w = x.size()
    x = x.reshape(b, t, c, h // block_size, block_size, w // block_size, block_size)
    x = x.permute(0, 1, 2, 4, 6, 3, 5)
    x = x.reshape(b, t, c * (block_size ** 2), h // block_size, w // block_size)
    return x

class SpaceToDepth(nn.Cell):
    '''
    class:space2depth
    '''
    def __init__(self, block_size):
        '''
        init
        '''
        super().__init__()
        self.bs = block_size

    def construct(self, x):
        '''forward'''
        return self.space2depth(self.bs, x)


class Degradation(nn.Cell):
    '''
    degradation branch for RefSR-NeRF
    '''
    def __init__(self, cur_mid_channels=32,
                cur_num_blocks=7, ref_mid_channels=32,
                ref_num_blocks=7, fus_mid_channels=32,
                fus_num_blocks=7, extr_channels=32, extr_num_blocks=7):

        super().__init__()

        self.cur_mid_channels = cur_mid_channels
        self.cur_num_blocks = cur_num_blocks
        self.ref_mid_channels = ref_mid_channels
        self.ref_num_blocks = ref_num_blocks
        self.fus_mid_channels = fus_mid_channels
        self.fus_num_blocks = fus_num_blocks

        mid_channels = cur_mid_channels
        self.mid_channels = mid_channels

        # propagation branches
        self.backward_resblocks_fus = ResidualBlocksWithInputConv(
            fus_mid_channels, fus_mid_channels, fus_num_blocks)

        self.backward_resblocks_sig = ResidualBlocksWithInputConv(
            ref_mid_channels, ref_mid_channels, ref_num_blocks)

        self.forward_resblocks_sig = ResidualBlocksWithInputConv(
            cur_mid_channels, cur_mid_channels, cur_num_blocks)

        self.sub_pixel_pos_diff = nn.Conv2d(12, mid_channels, 3, 1, 1)#input, output, 5 blocks

        self.conv_hrbg_res = ResidualBlocksWithInputConv(mid_channels, mid_channels, 2)
        self.feat_extract = nn.SequentialCell(
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, has_bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(extr_channels, extr_channels, extr_num_blocks),
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, has_bias=True))
        self.sd = SpaceToDepth(block_size=2)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels, mid_channels, 1, 1, 0, has_bias=True)
        self.upsample1 = nn.Conv2dTranspose(mid_channels, mid_channels,
                                            2, stride=2, padding=0,
                                            pad_mod='valid')
        self.upsample2 = PixelShufflePack(
            mid_channels, 3, 2, upsample_kernel=3)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def construct(self, lrs):
        '''forward'''
        n, t, c, h, w = lrs.size()

        lrs_m = self.sd(lrs)
        lrs_f = self.sub_pixel_pos_diff(lrs_m.view(-1, c*4, h//2, w//2))
        lrs_temp = lrs_f.view(n, t, self.mid_channels, h//2, w//2)
        lrs_f = self.feat_extract(lrs_f)
        lrs_f = lrs_f.view(n, t, self.mid_channels, h//2, w//2)
        h = h//2
        w = w//2

        # backward-time propgation
        outputs = []
        feat_prop = lrs_f.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat_prop = lrs_f[:, i, :, :, :] + feat_prop
                feat_prop = self.backward_resblocks_fus(feat_prop)
            else:
                feat_prop = lrs_f[:, i, :, :, :]
                feat_prop = self.backward_resblocks_sig(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = ops.ZerosLike(feat_prop)

        lr_currh = lrs_temp[:, 0, :, :, :]

        feat_prop = lrs_f[:, 0, :, :, :]
        feat_prop = self.forward_resblocks_sig(feat_prop)
        out = outputs[0] + feat_prop
        out = self.lrelu(self.fusion(out))
        out = lr_currh + out
        out = self.conv_hrbg_res(out)
        out = self.upsample2(out)
        base = lrs[:, 0, :, :, :]
        out += base
        outputs_final = []
        outputs_final.append(out)
        return ops.stack(outputs_final, dim=1)
    