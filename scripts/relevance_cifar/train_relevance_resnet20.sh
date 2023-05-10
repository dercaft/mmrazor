#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet20_cifar10.py
CHECKPOINT=${root}/checkpoints/resnet20_cifar10_9290.pth
#
# function=test_search_geatpy_discrete_inference
echo $PYTHONPATH
export PYTHONPATH=${root}
date=`date +%y%m%d_%H%M%S`

fullfilename=$(basename $0)
filename=${fullfilename%.*}
work_dir=/data/work_dirs/wyh/hybrid/${filename}_${date}

# CUDA_VISIBLE_DEVICES=$1 python ${root}/tools/mmcls/search_model.py \
#     ${CONFIG} \
#     --checkpoint_model ${CHECKPOINT} \
#     --work-dir ${work_dir} \
#     --search_function ${function} \
# # 需要指定路径
# json文件的路径
MODEL_FILE_PATH=$2
MODEL_INDEX=$1
#
# CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet18_cifar10_search.py
# CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet18_cifar10_train.py

CUDA_VISIBLE_DEVICES=$3 python ${root}/tools/mmcls/train_sample.py \
    $CONFIG \
    --work-dir /data/work_dirs/wyh/hybrid_train/${filename}_${date} \
    --model_index ${MODEL_INDEX} \
    --model_file_path ${MODEL_FILE_PATH} \
    --seed 0
