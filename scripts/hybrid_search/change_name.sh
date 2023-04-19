#!/bin/sh

source /etc/profile
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
for file in `ls $local`

do
    # echo `echo $local/$file | sed 's#run_hybrid_##g'`
    mv $local/$file `echo $local/$file | sed 's#run_hybrid_##g'`
done