import random
import sys, os
sys.path.append(os.pardir)
import json
import pandas as pd

root_path = "PGM_proposals"
dirs = os.listdir(root_path)
for csv in dirs:
    path = os.path.join(root_path, csv)
    id = csv[2:-4]
    dis = preprocess(path, id)
obj = preprocess(meta)

submission = {"version": "VERSION 1.3", "results": obj, "external_data": {"used": True, "details": "Excluding the last fc layer, the video encoding model (3D-ResneXt-101) is pre-trained on the Kinetics-400 training set"}}

with open("template.json", "w+") as f:
    json.dump(submission, f)
print("end creation of json file, saved at {}".format("template.json"))

def preprocess(csvpath, id):
    df = pd.read_csv(csvpath)
    data = []
    tmp = {}
    for idx in range(df.shape[0]):
        data = df.iloc[idx, 0:2]
        for xy in data:
            print(xy)
    # random generation of regions
    return None

