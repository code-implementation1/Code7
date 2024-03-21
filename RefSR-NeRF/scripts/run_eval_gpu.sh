#!/bin/bash
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

if [ $# != 5 ]
then
    echo "Usage: bash run_eval_gpu.sh [DATA_PATH] [DATA_CONFIG] [CHECKPOINT_PATH] [OUT_PATH] [RAYS_BATCH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_PATH=$(get_real_path $1)
DATA_CONFIG=$(get_real_path $2)
CHECKPOINT_PATH=$(get_real_path $3)
OUT_PATH=$(get_real_path $4)
RAYS_BATCH=$5


if [ ! -d $DATA_PATH ]
then
    echo "error: DATASET_PATH=$DATA_PATH is not a directory"
exit 1
fi

if [ ! -f $CHECKPOINT_PATH ]
then
    echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file"
exit 1
fi

echo "data_config: "$DATA_CONFIG
echo "dataset_path: "$DATA_PATH
echo "checkpoint_path: "$CHECKPOINT_PATH
echo "out_path: "$OUT_PATH
echo "rays_batch: "$RAYS_BATCH

python eval.py --data-path $DATA_PATH --data-config $DATA_CONFIG --model-ckpt $CHECKPOINT_PATH \
--out-path $OUT_PATH --rays-batch $RAYS_BATCH  &> eval_gpu.log &
