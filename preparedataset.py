import sys, os
import json
import time
import argparse

from pytube import YouTube


def urlgenerator(obj):
    for i in obj:
        for key, value in obj.items():
            if (key == 'database'):
                for key2, value2 in value.items():
                    yield key2, value2['url']

def downloader(urlgen):
    cnt = 0
    failcnt = 0
    succeedcnt = 0
    before = time.time()
    for id, url in urlgen:
        try:
            yt = YouTube(url)
            st = yt.streams.filter(file_extension='mp4').first()
            st.download(path)
            succeedcnt += 1
        except:
            failcnt += 1
            print('could not download id: {}, failcount = {}'.format(id, failcnt), flush=True)
            continue
        if succeedcnt % 5 == 4:
            elapsed = time.time() - before
            print('{} videos done, {}s per loop'.format(succeedcnt+1, elapsed/5), flush=True)
            before = time.time()
        cnt += 1
    print('complete, {}/{} videos in total'.format(succeedcnt, cnt), flush=True)


def main(args):
    path = os.path.join(args.dset_root, 'videos/')
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, args.json_path), 'r') as f:
        obj = json.load(f)
    downloader(urlgenerator(obj))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_root', '-d', type=str, default='~/ssd1/dsets/activitynet_captions')
    parser.add_argument('--json_path', '-j', type=str, default='activity_net.v1-3.min.json')
    args = parser.parse_args()
    main(args)

