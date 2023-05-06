#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet56_cifar10.py
CHECKPOINT=${root}/checkpoints/resnet56_cifar10_9403.pth
#
function=test_search_geatpy_discrete_inference
echo $PYTHONPATH
export PYTHONPATH=${root}
date=`date +%y%m%d_%H%M%S`

fullfilename=$(basename $0)
filename=${fullfilename%.*}
work_root_dir=/data/work_dirs/wyh/hybrid_ablation/${filename}_fixtotal_pool

total_size=2000

pool_size=500
max_epoch=`expr ${total_size} / ${pool_size}`
work_dir=${work_root_dir}_${pool_size}
CUDA_VISIBLE_DEVICES=$1 python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir ${work_dir} \
    --search_function ${function} \
    --cfg-options searcher.candidate_pool_size=${pool_size} searcher.max_epoch=${max_epoch}\

MODEL_FILE_PATH=${work_dir}/opt.json
MODEL_INDEX=0
CUDA_VISIBLE_DEVICES=$1 python ${root}/tools/mmcls/train_sample.py \
    $CONFIG \
    --work-dir ${work_dir} \
    --model_index ${MODEL_INDEX} \
    --model_file_path ${MODEL_FILE_PATH} \
    --seed 0
