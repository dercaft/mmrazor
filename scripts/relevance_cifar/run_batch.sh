#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

device=$1
bias=$2
# 用excel自动生成然后复制粘贴过来

i1=`expr ${device} \* 3 + ${bias} + 0`
i2=`expr ${device} \* 3 + ${bias} + 1`
i3=`expr ${device} \* 3 + ${bias} + 2`

bash ${local}/run_cifar_resnet_rel_train.sh $i1 $device

