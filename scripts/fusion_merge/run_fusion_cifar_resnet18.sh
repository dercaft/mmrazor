#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/pruning/fusion/cifar10/resnet18_cifar10_fusion.py
CHECKPOINT=${root}/checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth
#
function=$1
echo $PYTHONPATH
export PYTHONPATH=${root}
date=`date +%m%d_%s`
date=${${date}:0-6}
fullfilename=$(basename $0)
filename=${fullfilename%.*}

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/fusion/${filename}_${date} \
    --search_function ${function} \
