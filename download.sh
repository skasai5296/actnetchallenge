#!/bin/bash


# args:
# -d : dataset root path. should include json file. will create directory videos and downloads videos there.
# -j : json file relative path. should be downloaded from activitynet

nohup python preparedataset.py > out.log &!
echo "finished downloading activitynet dataset!"
