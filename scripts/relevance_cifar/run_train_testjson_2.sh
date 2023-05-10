#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

bash $local/train_relevance_resnet20.sh 50 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 51 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 52 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 53 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 54 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 55 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 56 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 57 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 58 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 59 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 60 ./relavent.json 2 
bash $local/train_relevance_resnet20.sh 61 ./relavent.json 2 
