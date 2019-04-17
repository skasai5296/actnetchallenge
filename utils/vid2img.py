import sys, os
import shutil
import cv2
import json

import argparse


# returns only the ids for videos in captions dataset
def get_actnetcaption_ids(idfiles):
    paths = []
    for file in idfiles:
        with open(file, 'r') as f:
            obj = json.loads(f.read())
            paths.extend(obj)
    # take off the "v_"
    return [id[2:]for id in paths]


# video_path    : where the video exists
# tar_path      : where the frames will be saved
# idfiles       : ids for videos to load
# shorter       : how many pixels for the shorter side of video

def vid2frames(video_path, tar_path, idfiles, shorter):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    loglist = []

    # Make the target directory if it doesn't exist.
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    idx = 0
    # for files in video directory,
    for filename in os.listdir(video_path):
        info = {}

        vid_id, ext = os.path.splitext(filename)
        if ext == '.mp4' and vid_id in idfiles:

            num_frames = 0
            cap = cv2.VideoCapture(os.path.join(video_path, filename))
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

            save_path = os.path.join(tar_path, vid_id)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # read frame and save until capture is done
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break
                # resize image
                image = cv2.resize(frame, dsize=(width, height))
                # frame number will be filled with zeros ex) 1 -> 000001
                tar = os.path.join(save_path, '{:06d}.jpg'.format(num_frames))
                # save
                cv2.imwrite(tar, image)
                num_frames += 1

            cap.release()

            # prepare metadata
            info['num_frames'] = num_frames
            info['video_id'] = vid_id
            info['index'] = idx
            info['framerate'] = fps
            info['width'] = width
            info['height'] = height

            print("Done with converting {} to frames, saved in {}".format(vid_id, save_path))
            print("{} frames in total, {} fps".format(num_frames, fps), flush=True)

            idx += 1

        loglist.append(info)

    return loglist

def main(args):
    idsdir = [os.path.join(args.rootdir, file) for file in args.idfiles]
    videodir = os.path.join(args.rootdir, args.videodir)
    savedir = os.path.join(args.rootdir, args.savedir)
    logfile = os.path.join(args.rootdir, args.logfile)

    idfiles = get_actnetcaption_ids(idsdir)

    if os.path.exists(args.logfile):
        sys.exit("log already exists")
    logs = vid2frames(videodir, savedir, idfiles, args.shorter)
    print("number of video files saved as frames: {}".format(len(logs)))
    with open(logfile, 'w+') as f:
        json.dump(logs, f)
    print("done!! Metadata in {}".format(logfile))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rootdir', type=str, default='../../../ssd1/dsets/activitynet_captions')
    parser.add_argument('--idfiles', type=list, default=['train_ids.json', 'val_ids.json', 'test_ids.json'])
    parser.add_argument('--videodir', type=str, default='videos')
    parser.add_argument('--savedir', type=str, default='frames')
    parser.add_argument('--logfile', type=str, default='videometa.json')
    parser.add_argument('--shorter', type=int, default=224)
    args = parser.parse_args()
    main(args)
