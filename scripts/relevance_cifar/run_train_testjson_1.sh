#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

bash $local/train_relevance_resnet20.sh 25 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 26 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 27 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 28 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 29 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 30 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 31 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 32 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 33 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 34 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 35 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 36 $local/relavent.json 1 
bash $local/train_relevance_resnet20.sh 37 $local/relavent.json 1 
