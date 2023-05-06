
# Path : /data/work_dirs/wyh/hybrid/run_cifar_20230413_2059/opt.json
# and type is : 
import os
import sys
import json
import csv
# Use first argument as dir path
DIR_PATH=sys.argv[1]
PREFIX=sys.argv[2]
# DIR_PATH="/data/work_dirs/wyh/cifar10_sample"
# list all dirs in DIR_PATH
dirs = os.listdir(DIR_PATH)
# traverse all dirs
for dir in dirs:
    if not dir.startswith(PREFIX):
        continue
    # list all files in dir
    files = os.listdir(os.path.join(DIR_PATH, dir))
    # traverse all files
    # name=""
    # cka=0
    # acc=0
    if not "opt.json" in files: continue
    opt=json.load(open(os.path.join(DIR_PATH, dir, "opt.json")))
    chacfg=opt[0]["channel_cfg"]
    rl=[]
    pc=[]
    rc=[]
    for n, cs in chacfg.items():
        # print("{n},\t{cs}".format(n=n,cs=cs))
        oc=cs["out_channels"]
        roc=cs["raw_out_channels"]
        r=oc/roc
        rl.append(r)
        pc.append(oc)
        rc.append(roc)
    print(dir,":")
    print("r:\t",rl)
    print("pc:\t",pc)
    print("rc:\t",rc)
    # write into csv
