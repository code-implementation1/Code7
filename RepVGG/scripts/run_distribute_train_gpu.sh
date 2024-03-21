#!/bin/bash
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
if [ $# != 3 ] && [ $# != 4 ]
then 
    echo "Usage: bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DATASET_PATH] [DEVICE_NUM] [PRETRAINED_CKPT_PATH](optional)"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

CONFIG_PATH=$(get_real_path "$1")
DATASET_PATH=$(get_real_path "$2")
DEVICE_NUM="$3"

if [ $# == 4 ]
then
    PRETRAINED_CKPT_PATH=$(get_real_path "$4")
fi

if [ ! -d "$DATASET_PATH" ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
    exit 1
fi


if [ $# == 4 ] && [ ! -f "$PRETRAINED_CKPT_PATH" ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PRETRAINED_CKPT_PATH is not a file"
    exit 1
fi

echo "config_file: $CONFIG_PATH"
echo "dataset_path: $DATASET_PATH"
echo "device_num: $DEVICE_NUM"
echo "pretrained: $PRETRAINED_CKPT_PATH"

ulimit -u unlimited

if [ $# == 3 ]
then
  mpirun --allow-run-as-root -n "$DEVICE_NUM" --output-filename log_output --merge-stderr-to-stdout \
         python train.py --config "$CONFIG_PATH" --device_num "$DEVICE_NUM" \
         --data_url "$DATASET_PATH" --device_target GPU &> dist_train.log &
fi

if [ $# == 4 ]
then
  mpirun --allow-run-as-root -n "$DEVICE_NUM" --output-filename log_output --merge-stderr-to-stdout \
        python train.py --config "$CONFIG_PATH" --device_num "$DEVICE_NUM" \
        --data_url "$DATASET_PATH" --pretrained "$PRETRAINED_CKPT_PATH" --device_target GPU &> dist_train.log &
fi