#!/bin/bash


# command line arguments of preparedataset.py
# -d : dataset root path. should include json file. will create directory videos and downloads videos there. REQUIREMENT
# -j : json file relative path. should be downloaded from activitynet. default is activity_net.v1-3.min.json
# -t : target path to save videos. default is 'videos' folder in root path.

nohup python preparedataset.py -d ~/hdd/dsets/activitynet_captions > out.log &!
