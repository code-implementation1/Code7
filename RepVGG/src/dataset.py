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
"""Create dataset and process dataset."""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter

from .augment.auto_augment import pil_interp, rand_augment_transform
from .augment.random_erasing import RandomErasing
from .augment.mixup import Mixup


class ImageNet:
    """ImageNet Define."""

    def __init__(self, args, training=True):
        """Init imagenet."""
        train_dir = os.path.join(args.data_url, 'train')
        val_dir = os.path.join(args.data_url, 'val')
        self.train_dataset = create_dataset_imagenet(
            train_dir, training=training, args=args
        )
        self.val_dataset = create_dataset_imagenet(
            val_dir, training=False, args=args
        )


def create_dataset_imagenet(dataset_dir, args, repeat_num=1, training=True):
    """Create a train or eval ImageNet2012 dataset for RepVGG.

    Parameters
    ----------
    dataset_dir: str
        The path of dataset.
    args: argparse.Namespace
        Parsed configuration and command line arguments.
    repeat_num: int
        The repeat times of dataset. Default: 1
    training: bool
        Whether dataset is used for train or eval.

    Returns
    -------
        Dataset
    """
    device_num, rank_id = _get_rank_info()
    shuffle = training
    cache_ds = None
    if args.cache_session_id and training:
        cache_ds = ds.DatasetCache(session_id=args.cache_session_id,
                                   prefetch_size=args.prefetch)
    if device_num == 1 or not training:
        if rank_id == 0 or rank_id is None:
            data_set = ds.ImageFolderDataset(
                dataset_dir, num_parallel_workers=args.num_parallel_workers,
                shuffle=shuffle, decode=True, cache=cache_ds
            )
        else:
            return None
    else:
        data_set = ds.ImageFolderDataset(
            dataset_dir,
            num_parallel_workers=args.num_parallel_workers, shuffle=shuffle,
            num_shards=device_num, shard_id=rank_id, decode=True,
            cache=cache_ds
        )

    transform = get_transforms(
        args.image_size, training=training,
        type=args.aug_type,
        interpolation=args.interpolation,
        auto_augment=args.auto_augment,
        re_prob=args.re_prob,
        re_mode=args.re_mode,
        re_count=args.re_count
    )

    transform_label = C.TypeCast(mstype.int32)

    data_set = data_set.map(
        input_columns='image',
        num_parallel_workers=args.num_parallel_workers,
        operations=transform,
    )
    data_set = data_set.map(
        input_columns='label',
        num_parallel_workers=args.num_parallel_workers,
        operations=transform_label
    )
    if (args.mix_up > 0. or args.cutmix > 0.) and not training:
        # if use mixup and not training(False), one hot val data label
        one_hot = C.OneHot(num_classes=args.num_classes)
        data_set = data_set.map(
            input_columns='label',
            num_parallel_workers=args.num_parallel_workers,
            operations=one_hot
        )
    # apply batch operations
    drop_remainder = training
    data_set = data_set.batch(
        args.batch_size if training else 1,
        drop_remainder=drop_remainder,
        num_parallel_workers=args.num_parallel_workers,
    )

    if (args.mix_up > 0. or args.cutmix > 0.) and training:
        mixup_fn = Mixup(
            mixup_alpha=args.mix_up,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=None,
            prob=args.mixup_prob,
            switch_prob=args.switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.label_smoothing,
            num_classes=args.num_classes
        )

        data_set = data_set.map(
            operations=mixup_fn,
            input_columns=['image', 'label'],
            num_parallel_workers=args.num_parallel_workers,
        )

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """Get rank size and rank id. It needs to define dataset."""
    rank_size = int(os.environ.get('RANK_SIZE', 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id


def get_transforms(
        image_size: int, training: bool, **aug: dict
):
    """Get images preprocessing according mode and augmentations settings.

    Parameters
    ----------
    image_size: int
        Target image size.
    training: bool
        Mode. If True augmentations may be applied.
    aug: Dict
        Augmentation settings (type, auto aug, random erase).

    Returns
    -------
        List of transforms.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    aug = {} if aug is None else aug
    if training:
        if aug['type'] == 'weak':
            transform = [
                vision.ToPIL(),
                vision.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                    interpolation=Inter.BILINEAR
                ),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.ToTensor(),
                vision.Normalize(mean, std, is_hwc=False),
            ]
        elif aug['type'] == 'none':
            transform = [
                vision.ToPIL(),
                vision.Resize(image_size, interpolation=Inter.BILINEAR),
                vision.CenterCrop(image_size),
                vision.ToTensor(),
                vision.Normalize(mean, std, is_hwc=False),
            ]
        elif aug['type'] == 'auto':
            aa_params = dict(
                translate_const=int(image_size * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in mean]),
                interpolation=pil_interp(aug['interpolation'])
            )
            auto_augment = aug['auto_augment']

            transform = [
                vision.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                    interpolation=Inter.BILINEAR
                ),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.ToPIL()
            ]
            if auto_augment is not None:
                transform += [rand_augment_transform(auto_augment, aa_params)]
            transform += [
                vision.ToTensor(),
                vision.Normalize(mean=mean, std=std, is_hwc=False),
                RandomErasing(
                    aug['re_prob'], mode=aug['re_mode'],
                    max_count=aug['re_count']),
            ]
        else:
            raise ValueError('???' + aug.get('type', 'Unknown'))
    else:
        transform = [
            vision.ToPIL(),
            vision.Resize(
                int((256 / 224) * image_size), interpolation=Inter.BILINEAR
            ),
            vision.CenterCrop(image_size),
            vision.ToTensor(),
            vision.Normalize(mean, std, is_hwc=False),
        ]

    return transform


def get_dataset(args, training=True):
    """Get model according to args.set."""
    datasets = {
        'ImageNet': ImageNet
    }
    dataset_type = 'ImageNet'
    print(f"=> Getting {dataset_type} dataset")
    dataset = datasets[dataset_type](args, training)

    return dataset
