import sys, os

vidpth = ["/ssd1/dsets/activitynet_captions/videos",
"/ssd1/dsets/activitynet_captions/frames",
"/ssd2/dsets/activitynet_captions/videos",
"/ssd2/dsets/activitynet_captions/frames"]

for root in vidpth:
    d = os.listdir(root)
    for path in d:
        print("v_"+path)
        os.rename(os.path.join(root, path), os.path.join(root, "v_" + path))
