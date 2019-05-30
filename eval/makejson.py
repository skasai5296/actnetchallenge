import random
import sys, os
sys.path.append(os.pardir)
import json

def preprocess(metadata):
    data = []
    lengths = [120, 240, 360, 480, 600, 720, 840, 1200]
    tmp = {}
    for obj in metadata:
        tmp['video_id'] = obj['video_id']
        tmp['sentence'] = None
        regs = []
        for duration in lengths:
            for i in range(10):
                begin = 0 if obj['num_frames'] <= duration else random.randrange(0, obj['num_frames']-duration)
                reg = [begin, min(obj['num_frames']-1, begin+duration)]
                regs.append(reg)
        tmp['timestamp'] = regs
        data.append(tmp)
    # random generation of regions
    return data

def process(submission)

temppath = "template.json"
if os.path.exists(temppath):
    with open(os.path.join(temppath), "r") as f:
        submission = json.load(f)
    obj = process(submission["results"])
else:
    with open(os.path.join("/ssd2/dsets/activitynet_captions", "videometa_test.json")) as f:
        meta = json.load(f)
    obj = preprocess(meta)

    submission = {"version": "VERSION 1.3", "results": obj, "external_data": {"used": True, "details": "Excluding the last fc layer, the video encoding model (3D-ResneXt-101) is pre-trained on the Kinetics-400 training set"}}

with open("template.json", "w+") as f:
    json.dump(submission, f)
print("end creation of json file, saved at {}".format("template.json"))
