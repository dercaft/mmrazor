import os
import sys

# Use first argument as dir path
DIR_PATH=sys.argv[1]

# DIR_PATH="/data/work_dirs/wyh/cifar10_sample"
# list all dirs in DIR_PATH
dirs = os.listdir(DIR_PATH)
# traverse all dirs
for dir in dirs:
    # list all files in dir
    files = os.listdir(os.path.join(DIR_PATH, dir))
    # traverse all files
    name=""
    cka=0
    acc=0
    for file in files:
        if file.endswith(".log"):
            name=file
    if name == "":
        continue
    with open(os.path.join(DIR_PATH, dir, name), "r") as f:
        lines = f.readlines()
        jump=True
        for i,line in enumerate(lines):
            if "space2ratio" in line:
                jump=False
                break
        if jump:
            continue
        for i,line in enumerate(lines):
            if "INFO - CKA:" in line:
                cka=float(lines[i+1].strip("[").strip("\n").strip("]"))
            if "Best accuracy_top-1" in line:
                # 2023-03-15 21:39:49,123 - mmcls - INFO - Best accuracy_top-1 is 89.8800 at 108 epoch.
                lacc=float(line.split(" ")[-4])
                if lacc>acc:
                    acc=lacc
    print("{cka},\t{acc}".format(cka=cka, acc=acc))