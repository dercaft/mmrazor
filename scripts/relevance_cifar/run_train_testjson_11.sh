#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
 
bash $local/train_relevance_resnet20.sh 38 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 39 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 40 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 41 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 42 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 43 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 44 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 45 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 46 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 47 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 48 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 49 ./relavent.json 1 