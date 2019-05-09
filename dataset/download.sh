#!/bin/bash


# command line arguments of preparedataset.py
# -d : dataset root path. should include json file. will create directory videos and downloads videos there. REQUIREMENT
# -j : json file relative path. should be downloaded from activitynet
# -t : target path to save videos. default is 'videos' folder in root path.

nohup python preparedataset.py -d /ssd2/dsets/activitynet_captions > out.log &!
