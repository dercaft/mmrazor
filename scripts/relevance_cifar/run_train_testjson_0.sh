#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

bash $local/train_relevance_resnet20.sh 0 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 1 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 2 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 3 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 4 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 5 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 6 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 7 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 8 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 9 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 10 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 11 ./relavent.json 0 
bash $local/train_relevance_resnet20.sh 12 ./relavent.json 0 