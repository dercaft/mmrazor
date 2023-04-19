#!/bin/bash
# -*- coding: utf-8 -*-
set -x
filep=$(dirname "$0") # file directory path
local=$(dirname $(dirname "$0")) # file directory path
root=$(dirname $(dirname $(dirname $local))) # project path

export PYTHONPATH=${root}

METHOD=CDP
REDUCTION_RATIO=0.5
FUNCTION=test_score_compress
# 需要指定路径
CONFIG=${root}/configs/pruning/simple_channel/imagenet/resnet50_8xb32_in1k.py
# configs/pruning/simple_channel/imagenet/resnet50_8xb32_in1k.py
CHECKPOINT=${root}/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth
# checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth
python ${local}/test_mmcls.py \
    $CONFIG \
    --checkpoint_model $CHECKPOINT \
    --metrics accuracy \
    --execute_function $FUNCTION \
    --reduction_ratio $REDUCTION_RATIO \
    --cfg-options algorithm.pruner.pruning_metric_name=${METHOD} \
    --save_compress_algo "$filep/resnet50_imageneet_$METHOD_$REDUCTION_RATIO.pth"