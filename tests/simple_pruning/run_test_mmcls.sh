#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

export PYTHONPATH=${root}

METHOD=CDP
REDUCTION_RATIO=0.5
FUNCTION=test_score_compress
# 需要指定路径
CONFIG=${root}/configs/pruning/simple_channel/cifar/resnet18_cifar10_search.py
CHECKPOINT=${root}/checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth
# 

python ${local}/test_mmcls.py \
    $CONFIG \
    --checkpoint_model $CHECKPOINT \
    --metrics accuracy \
    --execute_function $FUNCTION \
    --reduction_ratio $REDUCTION_RATIO \
    --cfg-options algorithm.pruner.pruning_metric_name=${METHOD} \
    --save_compress_algo "./resnet18_cifar10_{$METHOD}_{$REDUCTION_RATIO}.pth"