#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/
CHECKPOINT=${root}/checkpoints/
#

python ${local}/test_mmcls.py \
    $CONFIG \
    --checkpoint_model $CHECKPOINT \
    --metrics accuracy \