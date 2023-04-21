#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
GPU_NUM=$1
# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/imagenet/resnet18_in1k_search.py
CHECKPOINT=${root}/checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
# 
function=test_search_geatpy_discrete_inference
echo $PYTHONPATH
export PYTHONPATH=${root}
date=`date +%y%m%d_%H%M%S`

fullfilename=$(basename $0)
filename=${fullfilename%.*}
work_dir=/data/work_dirs/wyh/hybrid/${filename}_${date}
CUDA_VISIBLE_DEVICES=$GPU_NUM python ${root}/tools/mmcls/search_model.py \
    ${CONFIG} \
    --checkpoint_model ${CHECKPOINT} \
    --work-dir ${work_dir} \
    --search_function ${function} \

date=`date +%y%m%d_%H%M%S`
fullfilename=$(basename $0)
filename=${fullfilename%.*}
# 以上为每个具体训练的脚本应有的共同开头
# 需要指定路径
# json文件的路径
MODEL_FILE_PATH=${work_dir}/opt.json
MODEL_INDEX=0

# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=$((`date +%H` % 1 + 1))`date +%M%S`
# PORT=${PORT:-29400}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES=$GPU_NUM python ${root}/tools/mmcls/train_sample.py \
    $CONFIG \
    --work-dir /data/work_dirs/wyh/hybrid_train/${filename}_${date} \
    --model_index ${MODEL_INDEX} \
    --model_file_path ${MODEL_FILE_PATH} \
    --seed 0
