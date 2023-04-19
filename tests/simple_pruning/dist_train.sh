#!/usr/bin/env bash

CONFIG="configs/pruning/simple_channel/cifar/resnet18_cifar10_train.py"
GPUS=2
RESUME="tests/simple_pruning/resnet18_cifar10_CDP_0.5.pth"
WORK_DIR="/data/work_dirs/wyh/"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29350}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_mmcls.py $CONFIG --work-dir $WORK_DIR --start_file_path $RESUME --launcher pytorch ${@:3}
