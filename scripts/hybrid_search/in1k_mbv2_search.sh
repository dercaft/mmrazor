#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/imagenet/mobilenetv2_search.py
CHECKPOINT=${root}/checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth
#
function=test_search_geatpy_discrete_inference
echo $PYTHONPATH
export PYTHONPATH=${root}

date=`date +%y%m%d_%H%M%S`

fullfilename=$(basename $0)
filename=${fullfilename%.*}

CUDA_VISIBLE_DEVICES=$1 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \