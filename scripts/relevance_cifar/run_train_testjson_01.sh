#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
 
bash $local/train_relevance_resnet20.sh 13 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 14 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 15 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 16 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 17 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 18 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 19 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 20 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 21 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 22 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 23 $local/relavent.json 0 
bash $local/train_relevance_resnet20.sh 24 $local/relavent.json 0