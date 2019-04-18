import sys, os, time
import json
import shutil


def get_actnetcaption_ids(idfiles):
    paths = []
    for file in idfiles:
        with open(file, 'r') as f:
            obj = json.loads(f.read())
            paths.extend(obj)
    # take off the "v_"
    return [id[2:]for id in paths]

ids = get_actnetcaption_ids(['../../../ssd1/dsets/activitynet_captions/val_ids.json', '../../../ssd1/dsets/activitynet_captions/test_ids.json'])

vid_dir = '../../../ssd1/dsets/activitynet_captions/videos'
frm_dir = '../../../ssd1/dsets/activitynet_captions/frames'

for id in ids:
    vddir = os.path.join(vid_dir, id, '.mp4')
    frdir = os.path.join(frm_dir, id)
    if os.path.exists(frdir):
        shutil.rmtree(frdir)
print('done')
