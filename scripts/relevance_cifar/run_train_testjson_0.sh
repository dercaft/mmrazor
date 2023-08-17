#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

bash $local/train_relevance_resnet20.sh 0 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 1 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 2 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 3 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 4 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 5 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 6 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 7 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 8 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 9 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 10 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 11 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 12 $local/relavent.json 0 