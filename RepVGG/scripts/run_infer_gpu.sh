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
if [ $# != 4 ]
then
    echo "Usage: bash scripts/run_infer_gpu.sh [CONFIG_FILE] [DATA_PATH] [OUTPUT_PATH] [CHECKPOINT_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

CONFIG_FILE="$1"
DATA_PATH=$(get_real_path "$2")
OUTPUT_PATH=$(get_real_path "$3")
CHECKPOINT_PATH=$(get_real_path "$4")

if [ ! -f "$CHECKPOINT_PATH" ]
then
    echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file"
exit 1
fi

echo "config_path: $CONFIG_FILE"
echo "dataset_path: $DATA_PATH"
echo "output_path: $OUTPUT_PATH"
echo "checkpoint_path: $CHECKPOINT_PATH"

python infer.py --dataset_path "$DATA_PATH" --config "$CONFIG_FILE" --pretrained "$CHECKPOINT_PATH" --pred_output "$OUTPUT_PATH" --device_target GPU&> infer_gpu.log &
