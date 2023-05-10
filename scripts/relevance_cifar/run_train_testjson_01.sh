#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
 
bash $local/train_relevance_resnet20.sh 13 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 14 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 15 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 16 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 17 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 18 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 19 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 20 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 21 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 22 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 23 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 24 ./relavent.json 0