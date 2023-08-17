#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
 
bash $local/train_relevance_resnet20.sh 88 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 89 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 90 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 91 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 92 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 93 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 94 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 95 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 96 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 97 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 98 $local/relavent.json 3 
bash $local/train_relevance_resnet20.sh 99 $local/relavent.json 3 