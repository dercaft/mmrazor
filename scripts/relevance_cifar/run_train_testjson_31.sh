#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
 
bash $local/train_relevance_resnet20.sh 88 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 89 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 90 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 91 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 92 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 93 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 94 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 95 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 96 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 97 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 98 ./relavent.json 3 
bash $local/train_relevance_resnet20.sh 99 ./relavent.json 3 