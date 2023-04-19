#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/imagenet/resnet18_in1k_search.py
CHECKPOINT=${root}/checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
# 
function=$1
echo $PYTHONPATH
export PYTHONPATH=${root}
date=`date +%y%m%d_%H%M%S`

fullfilename=$(basename $0)
filename=${fullfilename%.*}

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
