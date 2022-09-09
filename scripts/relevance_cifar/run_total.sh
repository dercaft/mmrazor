#!/bin/bash
# -*- coding: utf-8 -*-
set -x
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

# 用excel自动生成然后复制粘贴过来
bash ${local}/run_batch.sh 0 ${bias} &