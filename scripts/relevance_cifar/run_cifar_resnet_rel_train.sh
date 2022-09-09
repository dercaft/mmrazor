#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
#
date=`date +%m%d_%s`
fullfilename=$(basename $0)
filename=${fullfilename%.*}
# 以上为每个具体训练的脚本应有的共同开头
# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet18_cifar10_train.py
CHECKPOINT=${root}/checkpoints/
MODEL_FILE_PATH=${root}/others/
MODEL_INDEX=$1
#
CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/train_sample.py \
    $CONFIG \
    --work-dir ${root}/work_dirs/${filename}_${date} \
    --model_index ${MODEL_INDEX} \
    --model_file_path ${MODEL_FILE_PATH} \
    --seed 0
    