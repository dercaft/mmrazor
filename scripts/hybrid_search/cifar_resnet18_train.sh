#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
#
date=`date +%y%m%d_%H%M%S`
fullfilename=$(basename $0)
filename=${fullfilename%.*}
# 以上为每个具体训练的脚本应有的共同开头
# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet18_cifar10_train.py
CHECKPOINT=${root}/checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth
# json文件的路径
MODEL_FILE_PATH=$2
MODEL_INDEX=$1
#
export PYTHONPATH=${root}

# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=$((`date +%H` % 1 + 1))`date +%M%S`
# PORT=${PORT:-29400}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES=$3 python ${root}/tools/mmcls/train_sample.py \
    $CONFIG \
    --work-dir /data/work_dirs/wyh/hybrid_train/${filename}_${date} \
    --model_index ${MODEL_INDEX} \
    --model_file_path ${MODEL_FILE_PATH} \
    --seed 0
    