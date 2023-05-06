#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29400}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

date=`date +%y%m%d_%H%M%S`
fullfilename=$(basename $0)
filename=${fullfilename%.*}
# 以上为每个具体训练的脚本应有的共同开头
# 需要指定路径
# json文件的路径
MODEL_FILE_PATH=${work_dir}/opt.json
MODEL_INDEX=0

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_mmcls.py \
    $CONFIG \
    --launcher pytorch ${@:3} \
    --work-dir /data/work_dirs/wyh/
