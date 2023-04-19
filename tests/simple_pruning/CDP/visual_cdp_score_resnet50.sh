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
CONFIG=${root}/configs/pruning/simple_channel/cifar/resnet50_search.py
CHECKPOINT=${root}/checkpoints/resnet50_b16x8_cifar10_20210528-f54bfad9.pth
# 
python ${local}/test_mmcls.py \
    $CONFIG \
    --checkpoint_model $CHECKPOINT \
    --metrics accuracy \
    --execute_function $FUNCTION \
    --reduction_ratio $REDUCTION_RATIO \
    --cfg-options algorithm.pruner.pruning_metric_name=${METHOD} \
    --save_compress_algo "$filep/resnet50_cifar10_$METHOD_$REDUCTION_RATIO.pth"