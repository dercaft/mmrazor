#!/bin/bash
# -*- coding: utf-8 -*-
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path

bash $local/train_relevance_resnet20.sh 50 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 51 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 52 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 53 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 54 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 55 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 56 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 57 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 58 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 59 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 60 $local/relavent.json 2 
bash $local/train_relevance_resnet20.sh 61 $local/relavent.json 2 
