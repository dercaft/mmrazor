#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet20_cifar10.py
CHECKPOINT=${root}/checkpoints/resnet20_cifar10_9290.pth
# CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet56_cifar10.py
# CHECKPOINT=${root}/checkpoints/resnet56_cifar10_9403.pth
#
#
function=test_from_json_channel_num_shift
# function=test_from_json_channel_num_shift_fix_group_050
# function=test_search_geatpy_discrete_inference
# function=test_search_geatpy_discrete_inference
echo $PYTHONPATH
export PYTHONPATH=${root}
date=`date +%y%m%d_%H%M%S`

fullfilename=$(basename $0)
filename=${fullfilename%.*}
work_dir=/data/work_dirs/wyh/hybrid/${filename}_${date}

CUDA_VISIBLE_DEVICES=$1 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir ${work_dir} \
    --search_function ${function} \

