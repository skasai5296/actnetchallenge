import random
import sys, os
sys.path.append(os.pardir)
import json
import pandas as pd

def get_prop(csvpath, meta, id):
    df = pd.read_csv(csvpath)
    data = []
    num = get_frmnum(meta, id)
    if num is None:
        return None, None
    for idx, single in df.iterrows():
        start, end = single['xmin'], single['xmax']
        prop = [int(start*num), int(end*num)]
        tmp = {"timestamp" : prop}
        data.append(prop)
    return "v_" + str(id), data
    # random generation of regions

def get_frmnum(meta, id):
    with open(meta, "r") as f:
        obj = json.load(f)
    dst = list(filter(lambda item: item['video_id'] == id, obj))
    if dst == []:
        return None
    else:
        return int(dst[0]['num_frames'])

root_path = "../../../Downloads/PGM_proposals"
meta = "../../../Downloads/videometa_test.json"
dirs = os.listdir(root_path)
data = {}
for i, csv in enumerate(dirs):
    path = os.path.join(root_path, csv)
    id = csv[2:-4]
    id, regs = get_prop(path, meta, id)
    if id is not None:
        data[id] = regs
    if i % 100 == 99:
        print("{} done".format(i+1), flush=True)

submission = {"version": "VERSION 1.3", "results": data, "external_data": {"used": True, "details": "Excluding the last fc layer, the video encoding model (3D-ResneXt-101) is pre-trained on the Kinetics-400 training set"}}

with open("proposals_test.json", "w+") as f:
    json.dump(submission, f)
print("end creation of proposals, saved at {}".format("proposals_test.json"))


