#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

bash $local/train_relevance_resnet20.sh 25 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 26 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 27 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 28 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 29 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 30 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 31 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 32 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 33 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 34 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 35 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 36 ./relavent.json 1 
bash $local/train_relevance_resnet20.sh 37 ./relavent.json 1 
