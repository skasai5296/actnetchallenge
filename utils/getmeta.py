import sys, os
import shutil
import cv2
import json

import argparse


# returns only the ids for videos in captions dataset
def get_actnetcaption_ids(idfile):
    paths = []
    with open(idfile, 'r') as f:
        obj = json.loads(f.read())
        paths.extend(obj)
    # take off the "v_"
    return [id[2:]for id in paths]


def vid2meta(video_dir, frame_dir, idsdir, save_file, shorter):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    ls_vid = os.listdir(video_dir)
    logs = []
    idx = 0
    for mp4file in ls_vid:
        vid_id, ext = os.path.splitext(mp4file)
        if vid_id in idsdir:
            info = {}
            num_frames = 0

            videopath = os.path.join(video_dir, mp4file)
            cap = cv2.VideoCapture(videopath)

            orig_width = cap.get(3)
            orig_height = cap.get(4)
            w_gt_h = orig_width >= orig_height

            if w_gt_h:
                scale = shorter / orig_height
                height = shorter
                width = int(orig_width * scale)
            else:
                scale = shorter / orig_width
                width = shorter
                height = int(orig_height * scale)

            # get framerate of video
            if int(major_ver) < 3:
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)

            cap.release()

            framepath = os.path.join(frame_dir, vid_id)
            num_frames = len(os.listdir(framepath))

            # prepare metadata
            info['video_id'] = vid_id
            info['index'] = idx
            info['num_frames'] = num_frames
            info['framerate'] = fps
            info['width'] = width
            info['height'] = height
            logs.append(info)
            idx += 1

            if idx % 100 == 99:
                print("{} done".format(idx+1), flush=True)

    with open(save_file, "w+") as f:
        f.write(json.dumps(logs))
    print("saved metafile to {}".format(save_file))

    return len(logs)


def main(args):
    idsdir = os.path.join(args.rootdir, args.idfile)
    videodir = os.path.join(args.rootdir, args.videodir)
    framedir = os.path.join(args.rootdir, args.framedir)
    logfile = os.path.join(args.rootdir, args.savefile)

    idfiles = get_actnetcaption_ids(idsdir)

    if os.path.exists(logfile):
        sys.exit("log already exists")
    num = vid2meta(videodir, framedir, idfiles, logfile, args.shorter)
    print("Done! number of data retrieved: {}".format(num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rootdir', type=str, default='../../../ssd1/dsets/activitynet_captions')
    parser.add_argument('--videodir', type=str, default='videos')
    parser.add_argument('--framedir', type=str, default='frames')
    parser.add_argument('--idfile', type=str, default='train_ids.json')
    parser.add_argument('--savefile', type=str, default='videometa_train.json')
    parser.add_argument('--shorter', type=int, default=224)
    args = parser.parse_args()
    main(args)
