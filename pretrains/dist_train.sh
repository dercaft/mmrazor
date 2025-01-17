#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=12345
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

date=`date +%y%m%d_%H%M%S`
fullfilename=$(basename $0)
filename=${fullfilename%.*}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$3 python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} \
    --work-dir /data/work_dirs/wyh/hybrid_pretrain/${filename}_${date} \
    # --auto-scale-lr