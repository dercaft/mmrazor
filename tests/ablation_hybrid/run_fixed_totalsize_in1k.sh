#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
# CONFIG=${root}/configs/pruning/hybrid/imagenet/resnet18_in1k_search.py
# CHECKPOINT=${root}/checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
CONFIG=${root}/configs/pruning/hybrid/imagenet/mobilenetv2_search.py
CHECKPOINT=${root}/checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth
#
function=test_search_geatpy_discrete_inference
echo $PYTHONPATH
export PYTHONPATH=${root}
date=`date +%y%m%d_%H%M%S`

fullfilename=$(basename $0)
filename=${fullfilename%.*}

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=1000 searcher.max_epoch=2 \
    # --cfg-options searcher.max_epoch=2 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=500 searcher.max_epoch=4 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=200 searcher.max_epoch=10 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=100 searcher.max_epoch=20 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=50 searcher.max_epoch=40 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=40 searcher.max_epoch=50 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=20 searcher.max_epoch=100 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=10 searcher.max_epoch=200 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=5 searcher.max_epoch=400 \

CUDA_VISIBLE_DEVICES=$2 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir /data/work_dirs/wyh/hybrid/${filename}_${date} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=2 searcher.max_epoch=1000 \
