#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

export PYTHONPATH=${root}

# 需要指定路径

CONFIG=${root}/configs/pruning/chmer/cifar10/resnet18_cifar10_chmer.py
CHECKPOINT=${root}/checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth
#

python ${local}/test_mmcls.py \
    $CONFIG \
    --checkpoint_model $CHECKPOINT \
    --metrics accuracy \
    --execute_function fusion_pipeline \
    --reduction_ratio 0.5