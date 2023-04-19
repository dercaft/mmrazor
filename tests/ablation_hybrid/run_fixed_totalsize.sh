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
    --cfg-options searcher.candidate_pool_size=1000 searcher.max_epoch=2 \
    # --cfg-options searcher.max_epoch=2 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=500 searcher.max_epoch=4 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=200 searcher.max_epoch=10 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=100 searcher.max_epoch=20 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=50 searcher.max_epoch=40 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=40 searcher.max_epoch=50 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=20 searcher.max_epoch=100 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=10 searcher.max_epoch=200 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=5 searcher.max_epoch=400 \

python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=2 searcher.max_epoch=1000 \
