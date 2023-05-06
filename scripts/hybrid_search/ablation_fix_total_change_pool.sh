# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 需要指定路径
CONFIG=${root}/configs/pruning/hybrid/cifar10/resnet56_cifar10.py
CHECKPOINT=${root}/checkpoints/resnet56_cifar10_9403.pth
#
# bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 2000 $1
# bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 1000 $1
# bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 500 $1
# bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 200 $1
# bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 100 $1
# bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 50 $1

bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 40 $1
bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 20 $1
bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 10 $1
bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 5 $1
bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 2 $1
bash ${local}/resnet56_fix_total.sh $CONFIG $CHECKPOINT 2000 1 $1