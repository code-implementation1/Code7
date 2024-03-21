# Contents

* [Contents](#contents)
    * [RepVGG Description](#repvgg-description)
        * [Model Architecture](#model-architecture)
        * [Dataset](#dataset)
    * [Environment Requirements](#environment-requirements)
    * [Quick Start](#quick-start)
        * [Prepare the model](#prepare-the-model)
        * [Run the scripts](#run-the-scripts)
    * [Script Description](#script-description)
        * [Script and Sample Code](#script-and-sample-code)
        * [Script Parameters](#script-parameters)
    * [Training](#training)
        * [Training Process](#training-process)
        * [Transfer Training](#transfer-training)
        * [Distribute training](#distribute-training)
    * [Export](#export)
        * [Export process](#export-process)
        * [Export result](#export-result)
    * [Evaluation](#evaluation)
        * [Evaluation Process](#evaluation-process)
            * [Evaluation on GPU](#evaluation-on-gpu)
            * [ONNX Evaluation](#onnx-evaluation)
        * [Evaluation result](#evaluation-result)
    * [Inference](#inference)
        * [Inference Process](#inference-process)
            * [Inference on GPU](#inference-on-gpu)
            * [Inference with ONNX](#inference-with-onnx)
        * [Inference result](#inference-result)
   * [Model Description](#model-description)
        * [Performance](#performance)
   * [Description of Random Situation](#description-of-random-situation)
   * [ModelZoo Homepage](#modelzoo-homepage)

## [RepVGG Description](#contents)

RepVGG is simple but powerful architecture of convolutional neural network,
which has a VGG-like inference-time body composed of nothing but a stack of
3x3 convolution and ReLU, while the training-time model has a multi-branch
topology. Such decoupling of the training-time and inference-time architecture
is realized by a structural re-parameterization technique so that the model is
named RepVGG.

[Paper](https://arxiv.org/abs/2101.03697):   Xiaohan Ding, Xiangyu Zhang,
Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun. Computer Vision and Pattern
Recognition (CVPR), 2021 (In press).

### [Model Architecture](#contents)

RepVGG is network that has a VGG-like inference-time body composed of nothing
but a stack of 3x3 convolution and ReLU, while the training-time model has a
multi-branch topology. Such decoupling of the training-time and inference-time
architecture is realized by a structural re-parameterization technique.
RepVGG and the methodology of re-parameterization have been used in
**YOLOv6** ([paper](https://arxiv.org/abs/2209.02976))  and
**YOLOv7** ([paper](https://arxiv.org/abs/2207.02696)).
RepVGG has 5 stages and conducts down-sampling via stride-2 convolution at the
beginning of a stage. Here we only show the first 4 layers of a specific stage.

There are presented follows implemented architectures:

* RepVGG-A0
* RepVGG-A1
* RepVGG-A2
* RepVGG-B0
* RepVGG-B1
* RepVGG-B1g2
* RepVGG-B1g4
* RepVGG-B2
* RepVGG-B2g2
* RepVGG-B2g4
* RepVGG-B3
* RepVGG-B3g2
* RepVGG-B3g4
* RepVGG-D2se (This is not reported in the CVPR paper)

The names of the architectures are explained as follows.

Below is  table with architectural specification of RepVGG. Here 2 × 64*a*
means stage2 has 2 layers each with 64a channels.

| Stage | Output size    | RepVGG-A            | RepVGG-B           |
| ------| -------------- | ------------------- | ------------------ |
| 1     | 112 × 112      | 1 × min(64, 64*a*)  | 1 × min(64, 64*a*) |
| 2     | 56 × 56        | 2 × 64*a*           | 4 × 64*a*          |
| 3     | 28 × 28        | 4 × 128*a*          | 6 × 128*a*         |
| 4     | 14 × 14        | 14 × 256*a*         | 16 × 256*a*        |
| 5     | 7 × 7          | 1 × 512*b*          | 1 × 512*b*         |

Below is table RepVGG models defined by multipliers *a* and *b*.

| Name       | Layers of each stage  | *a*  | *b*   |
| -----------| --------------------- | ---- | ----- |
| RepVGG-A0  | 1, 2, 4, 14, 1        | 0.75 | 2.5   |
| RepVGG-A1  | 1, 2, 4, 14, 1        | 1    | 2.5   |
| RepVGG-A2  | 1, 2, 4, 14, 1        | 1.5  | 2.75  |
| RepVGG-B0  | 1, 4, 6, 16, 1        | 1    | 2.5   |
| RepVGG-B1  | 1, 4, 6, 16, 1        | 2    | 4     |
| RepVGG-B2  | 1, 4, 6, 16, 1        | 2.5  | 5     |
| RepVGG-B3  | 1, 4, 6, 16, 1        | 3    | 5     |

*g* in architecture name means that groupwise convolution is used instead
standard (`groups=1`).
*g2* means that `groups=2` and *g4* means that `groups=4`.

### [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

Dataset used: [ImageNet2012](http://www.image-net.org/)

* Dataset size: 146.6G
    * Train: 139.3G，1281167 images
    * Val: 6.3G，50000 images
    * Annotations: label of image is defined by image folder
* Data format: images sorted by label folders
    * Note: Data will be processed in dataset.py

## [Environment Requirements](#contents)

* Install [MindSpore](https://www.mindspore.cn/install/en).
* Download the dataset ImageNet dataset.
* Install third-parties requirements:

```text
numpy==1.21.6
onnxruntime-gpu==1.13.1
PyYAML==6.0
matplotlib==3.5.3
Pillow==9.2.0
tqdm==4.64.1
```

* We use ImageNet2012 as training dataset in this example by default, and you
 can also use your own datasets. For ImageNet-like dataset the directory
 structure is as follows:

```shell
.
└── imagenet
    ├── train
    │   ├── class1
    │   │    ├── 000000000001.jpg
    │   │    ├── 000000000002.jpg
    │   │    └── ...
    │   ├── ...
    │   └── classN
    ├── val
    │   ├── class1
    │   ├── ...
    │   └── classN
    └── test
```

## [Quick Start](#contents)

### [Prepare the model](#contents)

1. Prepare yaml config file. Create file and copy content from
 `src/configs/RepVGG.yaml` to created file.
1. Change the architecture name (`arch`) path to dataset (`data_url`), path to
 experiments folder (`train_url`), epochs number (`epochs`) and other additional
 hyper-parameters in created config.
1. Change other training hyper parameters (learning rate, regularization,
 augmentations etc.).
1. If you want to reproduce our experiments with RepVGG-B0, copy contents from
 `src/configs/RepVGG-B0_experiment.yaml` to config.

### [Run the scripts](#contents)

After installing MindSpore via the official website, you can start training and
evaluation as follows:

* running on GPU

```shell
# distributed training on GPU
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DATASET_PATH] [DEVICE_NUM] [PRETRAINED_CKPT_PATH](optional)

# standalone training on GPU
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [DATASET_PATH] [DEVICE_ID] [PRETRAINED_CKPT_PATH](optional)

# run eval on GPU
bash scripts/run_eval.sh [CONFIG_FILE] [DATASET_PATH] [CHECKPOINT_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─ cv
  └─ repvgg
    ├─ README.md                       ## descriptions about RepVGG
    ├─ scripts
    |  ├─ run_distribute_train_gpu.sh  ## bash script for distributed on gpu
    |  ├─ run_eval_gpu.sh              ## bash script for eval on gpu
    |  ├─ run_eval_onnx.sh             ## bash script for onnx model evaluation
    |  ├─ run_infer_gpu.sh             ## bash script for pred on gpu
    |  ├─ run_infer_onnx.sh            ## bash script for predictions using ONNX
    |  └─ run_standalone_train_gpu.sh  ## bash script for distributed on gpu
    ├─ src
    |  ├── __init__.py                 ## init file
    |  ├── config.py                   ## global args for repvgg
    |  ├── dataset.py                  ## dataset utilities
    |  ├── repvgg.py                   ## repvgg model define
    |  ├── augment
    |  │   ├── __init__.py             ## init file
    |  │   ├── auto_augment.py         ## auto_augment, rand_augment, and aug_mix
    |  │   ├── mixup.py                ## mixup and cutmix data preprocessing
    |  │   └── random_erasing.py       ## random erasing augmentation
    |  ├── configs
    |  │   ├── RepVGG.yaml                   ## standard config
    |  │   └── RepVGG-B0_experiment.yaml     ## reproduce config
    |  ├── ema
    |  │   ├── __init__.py                   ## init file
    |  │   ├── ema.py                        ## EMA functional
    |  │   └── train_one_step_with_ema.py    ## EMA specified TrainOneStep cell
    |  └── tools
    |      ├── __init__.py                   ## init file
    |      ├── amp.py                        ## AMP functional
    |      ├── callback.py                   ## custom callbacks
    |      ├── criterion.py                  ## optimized criterion utilities
    |      ├── optimizer.py                  ## optimizer creation
    |      ├── schedulers.py                 ## learning rate scheduler utilities
    |      └── utils.py                      ## additional tools
    ├── __init__.py                        ## init file
    ├── eval.py                            ## eval script (ckpt)
    ├── eval_onnx.py                       ## eval script (onnx)
    ├── export.py                          ## export mindir, onnx, air script
    ├── infer.py                           ## inference script (ckpt and mindir)
    ├── infer_onnx.py                      ## inference script (onnx)
    ├── requirements.txt                   ## list of requirements
    ├── train.py                           ## train scripts
    └── vis_img_preprocessing.py           ## image augmentation visualization
```

### [Script Parameters](#contents)

Major parameters in the yaml config file as follows:

```shell
# ===== Main settings ===== #
train_url: ''                  # Path to the folder
data_url: ''                   # Path to folder with dataset

# ===== Continue training or model inference ===== #
exclude_epoch_state: 0         # not use epoch state from pretraines checkpoint
pretrained: null               # path pretrained checkpoint
continues: null                # path to folder with checkpoint

# ===== Checkpoints setting ===== #
keep_checkpoint_max: 20        # max number of saved checkpoints (n last)
save_every: 1000               # interval (in steps) between checkpoints saving
keep_best_checkpoints_max: 5   # max number of saved checkpoints (n best)


# ===== Model architecture ===== #
arch: RepVGG-B0                # architecture name
num_classes: 1000              # number of classes
image_size: 224                # image size
deploy: 0                      # whether create model in deploy mode

# ===== Training duration settings ===== #
epochs: 120                    # number of training epoch
batch_size: 32                 # batch size
start_epoch: 0                 # manual start epoch number (useful on restarts)
start_step: 0                  # manual step of start epoch number

# ===== Optimizer settings ===== #
optimizer: momentum            # optimizer type
warmup_length: 0               # number of warmup epochs
warmup_lr: 0.00000007          # warmup learning rate
base_lr: 0.1                   # initial learning rate
min_lr: 0.0                    # minimal learning rate
momentum: 0.9                  # momentum coefficient for momentum optimizer
beta: [0.9, 0.999]             # meta for optimizer
lr_scheduler: cosine_lr        # scheduler for the learning rate
lr_adjust: 30                  # interval to drop learning rate (multistep_lr)
lr_gamma: 0.97                 # multi-step multiplier (multistep_lr)
weight_decay: 0.0001           # weight decay regularization coefficient
eps: 0.00000001                # adamw optimizer parameter

# ===== Train one step parameters (EMA and loss scale) ===== #
loss_scale: 1024               # loss scale
is_dynamic_loss_scale: 1       # whether loss scale is dynamic
with_ema: 0                    # whether use EMA (Exponential Moving Average)
ema_decay: 0.9999              # EMA decay parameter

# ===== Model optimization ===== #
amp_level: O0                  # AMP optimization level

# ===== Hardware settings ===== #
device_target: GPU             # platform (device) name
device_num: 8                  # number of devices
device_id: 0                   # device id

# ===== Data loading setting ===== #
num_parallel_workers: 4        # number of data loading workers
prefetch: 15                   # number of prefetched batches per worker
cache_session_id: null         # session id for cache admin server
use_data_sink: 0               # use data sink

# ===== Data augmentation settings ===== #
aug_type: weak                 # augmentation type

auto_augment: null             # auto augmetation definition
interpolation: bilinear        # auto augmentation interpolation type
re_prob: 0.0                   # random erasing probability (auto augmentation)
re_mode: pixel                 # filling type of erasing (randorm erashing)
re_count: 1                    # number of erased fields (randorm erashing)

mixup_prob: 0.0                # probability of applying mixup or cutmix
mix_up: 0.0                    # mixup alpha value
cutmix: 0.0                    # cutmix alpha value
switch_prob: 0.0               # probability of switching to mixup->cutmix
mixup_mode: batch              # how to apply params ("batch", "pair" "elem")

label_smoothing: 0.0           # label smoothing to use

# ===== Other ===== #
brief: epochs-120_bs-32x8_lr-0.1-0.0    # brief model definition
seed: 0                                 # random seed
 ```

Set number of epochs may not be enough to obtain target accuracy.
It is proposed to continue training with the following updated parameters:

* `epochs`: (set number of epochs, for example, 50)
* `base_lr`: (set base learning rate, for example, 0.0005)
* `pretrained`: (set path to obtained checkpoint repvgg/models/path/to/best/ckpt)
* `exclude_epoch_state`: 1

## [Training](#contents)

To train the model, run `train.py`.

### [Training process](#contents)

Standalone training mode:

```bash
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [DATASET_PATH] [DEVICE_ID] [PRETRAINED_CKPT_PATH](optional)
```

We need several parameters for these scripts.

* `CONFIG_PATH`: parameter configuration.
* `DATASET_PATH`: the path of ImageNet-like dataset.
* `DEVICE_ID`: need to define used device.
* `PRETRAINED_CKPT_PATH`: the path of pretrained checkpoint file, it is better
 to use absolute path.

Training result will be stored in the current path, whose folder name is "LOG".
Under this, you can find checkpoint files together with result like the
followings in log.

```log
epoch: 1 step: 1000, loss is 6.1155386
epoch: 1 step: 2000, loss is 5.2587314
epoch: 1 step: 3000, loss is 4.766342
epoch: 1 step: 4000, loss is 4.6344023
epoch: 1 step: 5000, loss is 4.2475777
Eval epoch time: 471432.161 ms, per step time: 9.429 ms
Result metrics epoch 1:  {'acc': 0.20744, 'eval_loss': 4.539827346801758, 'loss': 3.9532574266518408}
Train epoch time: 1501689.509 ms, per step time: 300.098 ms
Eval result: epoch 1, metrics: {'acc': 0.20744, 'loss': 3.9532574266518408}
epoch: 2 step: 996, loss is 3.824325
epoch: 2 step: 1996, loss is 3.5784812
epoch: 2 step: 2996, loss is 4.7575254
epoch: 2 step: 3996, loss is 3.280191
epoch: 2 step: 4996, loss is 3.4426632
Eval epoch time: 471680.317 ms, per step time: 9.434 ms
Result metrics epoch 2:  {'acc': 0.32602, 'eval_loss': 3.5014808177948, 'loss': 3.158886290922643}
Train epoch time: 1250603.395 ms, per step time: 249.921 ms
Eval result: epoch 2, metrics: {'acc': 0.32602, 'loss': 3.158886290922643}
...
epoch: 120 step: 524, loss is 0.67218107
epoch: 120 step: 1524, loss is 1.0175849
epoch: 120 step: 2524, loss is 1.1731689
epoch: 120 step: 3524, loss is 1.572964
epoch: 120 step: 4524, loss is 1.039779
Eval epoch time: 470393.290 ms, per step time: 9.408 ms
Result metrics epoch 120:  {'acc': 0.7475, 'eval_loss': 2.167008876800537, 'loss': 1.0252961638786127}
Train epoch time: 1285104.409 ms, per step time: 256.815 ms
Eval result: epoch 120, metrics: {'acc': 0.7475, 'loss': 1.0252961638786127}
train success
```

```log
epoch: 1 step: 1000, loss is 0.8249574
epoch: 1 step: 2000, loss is 1.0713954
epoch: 1 step: 3000, loss is 1.7736989
epoch: 1 step: 4000, loss is 1.6329188
epoch: 1 step: 5000, loss is 0.9781703
Eval epoch time: 301740.919 ms, per step time: 6.035 ms
Result metrics epoch 1:  {'acc': 0.7455, 'eval_loss': 1.843936800956726, 'loss': 1.0268876999970082}
Train epoch time: 1115606.350 ms, per step time: 222.943 ms
Eval result: epoch 1, metrics: {'acc': 0.7455, 'loss': 1.0268876999970082}
...
epoch: 36 step: 860, loss is 0.886052
epoch: 36 step: 1860, loss is 1.0655422
epoch: 36 step: 2860, loss is 1.5963659
epoch: 36 step: 3860, loss is 1.5159011
epoch: 36 step: 4860, loss is 0.76063764
Eval epoch time: 303844.624 ms, per step time: 6.077 ms
Result metrics epoch 36:  {'acc': 0.75056, 'eval_loss': 1.829211950302124, 'loss': 1.0140869947769366}
Train epoch time: 1132979.781 ms, per step time: 226.415 ms
Eval result: epoch 36, metrics: {'acc': 0.75056, 'loss': 1.0140869947769366}
...
epoch: 50 step: 804, loss is 0.84612405
epoch: 50 step: 1804, loss is 1.3933551
epoch: 50 step: 2804, loss is 0.92925334
epoch: 50 step: 3804, loss is 0.65676713
epoch: 50 step: 4804, loss is 0.8333833
Eval epoch time: 281207.215 ms, per step time: 5.624 ms
Result metrics epoch 50:  {'acc': 0.75022, 'eval_loss': 2.1325695514678955, 'loss': 1.0113875038196734}
Train epoch time: 1109067.105 ms, per step time: 221.636 ms
Eval result: epoch 50, metrics: {'acc': 0.75022, 'loss': 1.0113875038196734}
```

### [Transfer Training](#contents)

You can train your own model based on either pretrained classification model
or pretrained detection model. You can perform transfer training by following
steps.

1. Convert your own ImageNet-like dataset. Otherwise, you have to add your own
 data preprocess code.
1. Change RepVGG.yaml according to your own dataset, especially the
 `num_classes`.
1. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by
 `pretrained` argument. Transfer training means a new training job, so just set
 `exclude_epoch_state` 1.
1. Build your own bash scripts using new config and arguments for further
 convenient.

### [Distribute training](#contents)

Distribute training mode:

```shell
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DATASET_PATH] [DEVICE_NUM] [PRETRAINED_CKPT_PATH](optional)
```

We need several parameters for this scripts:

* `CONFIG_PATH`: parameter configuration.
* `DATASET_PATH`: the path of ImageNet-like dataset.
* `DEVICE_NUM`: number of devices.
* `PRETRAINED_CKPT_PATH`: the path of pretrained checkpoint file, it is better
 to use absolute path.

## [Export](#contents)

### [Export process](#contents)

* Export model.

```shell
python export.py --pretrained [CKPT_PATH] --file_format [FILE_FORMAT] --export_path [EXPORT_PATH] --config [CONFIG_PATH]
```

The pretrained parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR", "ONNX"] (Currently export to air
format is not checked)

### [Export result](#contents)

Exporting result is file in ONNX or MINDIR format.

## [Evaluation](#contents)

### [Evaluation process](#contents)

#### [Evaluation on GPU](#contents)

```shell
bash scripts/run_eval_gpu.sh [CONFIG_FILE] [DATASET_PATH] [CHECKPOINT_PATH]
```

We need four parameters for this scripts.

* `CONFIG_PATH`: parameter configuration.
* `DATASET_PATH`: the absolute path for dataset subset (validation).
* `CHECKPOINT_PATH`: path to checkpoint (it may be path to MINDIR file).

> checkpoint can be produced in training process.

#### [ONNX Evaluation](#contents)

Run ONNX evaluation from repvgg directory:

 ```bash
bash scripts/run_eval_onnx.sh [ONNX_PATH] [DATASET_PATH] [DEVICE_TARGET(optional)]
```

We need three parameters for this scripts.

* `ONNX_PATH`： path to saved onnx model.
* `DATASET_PATH`: path to ImageNet subset folder.
* `DEVICE_TARGET`: computation platform (GPU or CPU).

### [Evaluation result](#contents)

Result for GPU:

```log
=> eval results:{'Loss': 1.0140895074831058, 'Top1-Acc': 0.75052, 'Top5-Acc': 0.92158}
=> eval success
```

Result for ONNX:

```log
=> eval results:{'Top1-Acc': 0.75052, 'Top5-Acc': 0.92158}
=> eval success
```

## [Inference](#contents)

### [Inference process](#contents)

#### [Inference on GPU](#contents)

Run model inference from repvgg directory:

```bash
bash scripts/run_infer_gpu.sh [CONFIG_FILE] [DATA_PATH] [OUTPUT_PATH] [CHECKPOINT_PATH]
```

We need 4 parameters for these scripts:

* `CONFIG_FILE`： path to config file.
* `DATA_PATH`: path to ImageNet subset folder.
* `OUTPUT_PATH`: path to output file.
* `CHECKPOINT_PATH`: path to checkpoint (it may be path to MINDIR file).

#### [Inference with ONNX](#contents)

Run model inference from repvgg directory:

```bash
bash scripts/run_infer_onnx.sh [ONNX_PATH] [DATASET_PATH] [OUTPUT_PATH] [DEVICE_TARGET(optional)]
```

We need 4 parameters for this scripts:

* `ONNX_PATH`： path to saved onnx model.
* `DATASET_PATH`: path to ImageNet subset folder.
* `OUTPUT_PATH`: path to output file.
* `DEVICE_TARGET`: computation platform (GPU or CPU).

### [Inference result](#contents)

Predictions will be output in logs and saved in JSON file. Predictions format
is same for mindspore and ONNX model File content is dictionary where key is
image path and value is class number. It's supported predictions for folder of
images (png, jpeg file in folder root) and single image.

Typical outputs of such script for single image:

```log
...
=====  Load single image  =====
/data/imagenet/val/n01440764/ILSVRC2012_val_00002138.JPEG (class: 0)
```

Typical outputs for folder with image:

```log
=====  Load directory  =====
/data/imagenet/val/n01440764/ILSVRC2012_val_00012503.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00024327.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00039905.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00017472.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00021740.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00028158.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00010306.JPEG (class: 133)
/data/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG (class: 391)
/data/imagenet/val/n01440764/ILSVRC2012_val_00009379.JPEG (class: 0)
...
/data/imagenet/val/n01440764/ILSVRC2012_val_00009346.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00017699.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00025527.JPEG (class: 0)
/data/imagenet/val/n01440764/ILSVRC2012_val_00024235.JPEG (class: 0)
```

## [Model Description](#contents)

### [Performance](#contents)

| Parameters          | GPU                                                                    |
| ------------------- | ---------------------------------------------------------------------- |
| Model Version       | RepVGG-B0                                                              |
| Resource            | 8xGPU(NVIDIA GeForce RTX 3090), CPU 2.1GHz 64 cores, Memory 256G       |
| Uploaded Date       | 12/07/2022 (month/day/year)                                            |
| MindSpore Version   | 1.9.0                                                                  |
| Dataset             | ImageNet2012                                                           |
| Training Parameters | epoch = 170 (120,lr=0.1->0 + 50,lr=0.0005->0),  batch_size = 32x8      |
| Optimizer           | Momentum                                                               |
| Loss Function       | SoftmaxCrossEntropyWithLogits                                          |
| Speed               | 8pcs: 253.824ms                                                        |
| Total time          | 8pcs: 60.35 (42.6h - first 120 epoch, 17.75 - follow 50 epochs)        |
| outputs             | accuracy                                                               |
| Accuracy            | 75.05%                                                                 |
| Model for inference | 15M (14.33M after re-parametrization)(.ckpt file)                      |
| configuration       | RepVGG-B0_experiment.yaml                                              |
| Scripts             | <https://gitee.com/mindspore/models/tree/master/official/cv/repvgg>    |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use
random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
