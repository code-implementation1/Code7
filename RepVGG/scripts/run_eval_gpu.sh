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
if [ $# != 3 ]
then
    echo "Usage: bash scripts/run_eval_gpu.sh [CONFIG_FILE] [DATASET_PATH] [CHECKPOINT_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

CONFIG_FILE=$(get_real_path "$1")
DATASET_PATH=$(get_real_path "$2")
CHECKPOINT_PATH=$(get_real_path "$3")


if [ ! -d "$DATASET_PATH" ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]
then
    echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file"
exit 1
fi

echo "config_file: $CONFIG_FILE"
echo "dataset_path: $DATASET_PATH"
echo "checkpoint_path: $CHECKPOINT_PATH"

python eval.py --config "$CONFIG_FILE" --dataset_path "$DATASET_PATH" --pretrained "$CHECKPOINT_PATH" --device_target GPU &> eval_gpu.log &

