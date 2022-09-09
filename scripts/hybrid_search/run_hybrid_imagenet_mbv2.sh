#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/imagenet/mobilenetv2_search.py
CHECKPOINT=${root}/checkpoints/
#
function=$1

date=`date +%m%d_%s`
fullfilename=$(basename $0)
filename=${fullfilename%.*}

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir ${root}/work_dirs/${filename}_${date} \
    --search_function ${function} \