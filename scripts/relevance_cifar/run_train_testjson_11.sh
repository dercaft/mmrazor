#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
 
bash $local/train_relevance_resnet20.sh 38 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 39 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 40 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 41 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 42 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 43 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 44 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 45 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 46 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 47 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 48 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 49 $local/relavent.json 1 