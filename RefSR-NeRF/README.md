
# Contents

* [Contents](#contents)
    * [NeRF Description](#nerf-description)
        * [Dataset](#dataset)
    * [Environment Requirements](#environment-requirements)
    * [Quick Start](#quick-start)
        * [Prepare the model](#prepare-the-model)
        * [Run the scripts](#run-the-scripts)
    * [Script Description](#script-description)
        * [Script and Sample Code](#script-and-sample-code)
        * [Script Parameters](#script-parameters)
    * [Evaluation](#evaluation)
        * [Evaluation Process](#evaluation-process)
            * [Evaluation on GPU](#evaluation-on-gpu)
        * [Evaluation result](#evaluation-result)
   * [Model Description](#model-description)
        * [Performance](#performance)
   * [Description of Random Situation](#description-of-random-situation)
   * [ModelZoo Homepage](#modelzoo-homepage)

## [RefSR-NeRF Description](#contents)

RefSR-NeRF is an end-to-end framework that first learns a low resolution NeRF representation, and then reconstructs the high frequency details with the help of a high resolution reference
image. We observe that simply introducing the pre-trained models from the literature tends to produce unsatisfied artifacts due to the divergence in the degradation model. To this
end, we design a novel lightweight RefSR model to learn the inverse degradation process from NeRF renderings to target HR ones. Extensive experiments on multiple benchmarks demonstrate that our method exhibits an impressive trade-off among rendering quality, speed, and memory usage, outperforming or on par with NeRF and its variants
while being 52× speedup with minor extra memory usage

[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_RefSR-NeRF_Towards_High_Fidelity_and_Super_Resolution_View_Synthesis_CVPR_2023_paper.pdf):
Xudong Huang, Wei Li, Jie Hu, Hanting Chen, Yunhe Wang . CVPR, 2023.

### [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in the original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

* LLFF (“Real Forward-Facing”): real images of complex scenes captured with roughly forward-facing images.
  This dataset consists of 8 scenes captured with a handheld cellphone (5 taken from the LLFF paper and 3 that we capture),
  captured with 20 to 62 images, and hold out 1/8 of these for the test set. All images are 1008×756 pixels.

## [Environment Requirements](#contents)

* Download the datasets and locate to some folder `/path/to/the/dataset`:

    * [llff](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7)

* Install the requirements:

Use requirement.txt file with the following command:

```bash
pip install -r requirements.txt
```

Or you can install the following packages manually:

```text
matplotlib
mindspore-dev==2.0.0.dev20230226
numpy
opencv-python
PyMCubes==0.1.2
scikit-learn
scipy
tqdm
trimesh==3.20.2
imageio[ffmpeg]
```

## [Quick Start](#contents)

### [Prepare the model](#contents)

All necessary configs examples can be found in the project directory `src/configs`.

Evaluation stage:

* dataset-based settings config (`*_ds_config.json`)
* nerf architecture config (the default in use)

Note: \* - dataset type (one from llff).

1. Prepare the model directory: copy necessary configs for choosing dataset to some directory `/path/to/model_cfgs/` for future scripts launching.
2. Hyper parameters that recommended to be changed for training stage in dataset-based train config:

    * `train_rays_batch_size`
    * `val_rays_batch_size`
    * `epochs`
    * `val_epochs_freq`
    * `lr`
    * `lr_decay_rate`
    * `precrop_train_epochs`

### [Run the scripts](#contents)

After preparing directory with configs `/path/to/model_cfgs/` you can start training and evaluation:

* running on GPU

```shell
# run eval on GPU
bash scripts/run_eval.sh [DATA_PATH] [DATA_CONFIG] [CHECKPOINT_PATH] [OUT_PATH] [RAYS_BATCH]

# run export 3d mesh
bash scripts/run_export_3d.sh [SCENE_CONFIG] [CHECKPOINT_PATH] [OUT_STL]

# run export video
bash scripts/run_export_video.sh [POSES] [SCENE_CONFIG] [CHECKPOINT_PATH] [OUT_PATH] [RAYS_BATCH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
├── cv
│ └── RefSR-NeRF
│    ├── convert_weights.py
│     ├── eval.py                                 ## evaluation script
│     ├── export_3d_model.py                      ## 3d model exporting
│     ├── README.md                               ## NeRF description
│     ├── requirements.txt                        ## requirements
│     ├── scripts
│     │    ├── run_eval_gpu.sh                    ## bash script for evaluation
│     │    ├── run_export_3d.sh                   ## bash script for 3d mesh exporting
│     │    ├── run_export_video.sh                ## bash script for video exporting
│     ├── src
│     │    ├── configs
│     │    │    ├── inference
│     │    │    │    ├── blend_poses.json         ## blender dataset one pose config
│     │    │    │    ├── blend_scene_config.json  ## blender dataset scene config
│     │    │    │    ├── blend_video_poses.json   ## blender dataset poses for video exporting
│     │    │    │    ├── llff_poses.json          ## LLFF dataset one pose config
│     │    │    │    ├── llff_scene_config.json   ## LLFF dataset scene config
│     │    │    │    └── llff_video_poses.json    ## LLFF dataset poses for video exporting
│     │    │    ├── llff_ds_config.json           ## LLFF dataset settings config
│     │    │    ├── llff_train_config.json        ## LLFF dataset train config
│     │    │    └── nerf_config.json              ## NeRF architecture config
│     │    ├── data
│     │    │    ├── data_loader.py                ## module with dataset loader func
│     │    │    ├── dataset.py                    ## mindspore based datasets
│     │    │    ├── ds_loaders
│     │    │    │    ├── llff.py                  ## LLFF dataset loader
│     │    │    └── __init__.py                   ## init file
│     │    ├── model
│     │    │    ├── deg.py                        ## module with degredation branch
│     │    │    ├── sr.py                         ## module of sr branch
│     │    │    ├── unet_parts.py                 #refine module
│     │    │    ├── unet.py                       #refine module
│     │    ├── __init__.py                        ## init file
│     │    ├── tools
│     │    │    ├── callbacks.py                  ## custom callbacks
│     │    │    ├── common.py                     ## auxiliary funcs
│     │    │    ├── __init__.py                   ## init file
│     │    │    ├── mlflow_funcs.py               ## mlflow auxiliary funcs
│     │    │    └── rays.py                       ## rays sampling
│     │    └── volume_rendering
│     │        ├── coordinates_samplers.py        ## NeRF coordinates sampling
│     │        ├── __init__.py                    ## init file
│     │        ├── scene_representation.py        ## NeRF scene representation
│     │        └── volume_rendering.py            ## NeRF volume rendering pipeline
```

### [Script Parameters](#contents)

Dataset settings config parameters differ based on the dataset. But the common
dataset settings:

```bash
{
  "data_type": "llff",                # dataset type - one from blender and llff
  "white_background": false,          # make white background after image loading - useful for the synthetic scenes
  "is_ndc": true,                     # normal device coordinates space rays sampling
  "linear_disparity_sampling": false  # linear points sampling along ray
}
```

## [Evaluation](#contents)

### [Evaluation process](#contents)

#### [Evaluation on GPU](#contents)

```bash
bash scripts/run_eval_gpu.sh [DATA_PATH] [DATA_CONFIG] [CHECKPOINT_PATH] [OUT_PATH] [RAYS_BATCH]
```

Script parameters:

* `DATA_PATH`: the path to the blender or llff dataset scene.
* `DATA_CONFIG`: dataset scene loading settings config.
* `CHECKPOINT_PATH`: the path of pretrained checkpoint file.
* `OUT_PATH`: output directory to store the evaluation result.
* `SCALE`: super resolution scale.
* `RAYS_BATCH`: rays batch number for NeRF evaluation. Should be the divider of image height * width.

### [Evaluation result](#contents)

Result:

* Store GT and predict images with the PSNR metric in the files name.
* Store the CSV file with image-wise PSNR value and the total PSNR.

## [Model Description](#contents)

### [Performance](#contents)

| Parameters          | GPU                                                     |
| ------------------- |---------------------------------------------------------|
| Model Version       | RefSR-NeRF                                              |
| Resource            | 1xGPU(NVIDIA V100), CPU 2.1GHz 64 cores, Memory 256G    |
| Uploaded Date       | 12/25/2023 (month/day/year)                             |
| MindSpore Version   | 2.0.0.dev20230226                                       |
| Dataset             | LLFF (realistic)                                        |

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
