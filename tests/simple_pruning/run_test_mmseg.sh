#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

export CUDA_VISIBLE_DEVICES=0
GPUS=1
PORT=${PORT:-29510}
# 需要指定路径
CONFIG=${root}/configs/
CHECKPOINT=${root}/checkpoints/
#
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ${local}/test_mmseg.py \
    $CONFIG \
    --checkpoint_model $CHECKPOINT \
    --launcher pytorch \
    --eval mIoU \