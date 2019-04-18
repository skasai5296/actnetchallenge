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
    fullflag = False

    # Make the target directory if it doesn't exist.
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    dirs = os.listdir(video_path)
    total = len(dirs)
    num = 0
    # for files in video directory,
    for i, filename in enumerate(dirs):
        info = {}

        vid_id, ext = os.path.splitext(filename)
        save_path = os.path.join(tar_path, vid_id)
        # if extension is mp4 and is in the ids to download, but doesn't already have a directory,

        if ext == '.mp4' and vid_id in idfiles and not os.path.exists(save_path):

            num_frames = 0
            os.mkdir(save_path)
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

            # read frame and save until capture is done
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break
                # resize image
                image = cv2.resize(frame, dsize=(width, height))
                # frame number will be filled with zeros ex) 1 -> 000001
                tar = os.path.join(save_path, '{:06d}.jpg'.format(num_frames))
                # try saving, failing will abort download (due to full storage?)
                try:
                    cv2.imwrite(tar, image)
                except:
                    fullflag = True
                    break
                num_frames += 1

            cap.release()

            # prepare metadata
            print("Done with converting {} to frames, saved in {}".format(vid_id, save_path))
            print("{} frames in total".format(num_frames), flush=True)
            num += 1

        if fullflag:
            print("Could not save, aborting")
            break
        if i % 1000 == 999:
            print("{}/{} done".format(i+1, total), flush=True)

    return fullflag

def main(args):
    idsdir = [os.path.join(args.rootdir, file) for file in args.idfiles]
    videodir = os.path.join(args.rootdir, args.videodir)
    savedir = os.path.join(args.rootdir, args.savedir)

    idfiles = get_actnetcaption_ids(idsdir)

    flag = vid2frames(videodir, savedir, idfiles, args.shorter)
    print("number of video files saved as frames: {}".format(num))
    if not flag:
        print("done!!")
    else:
        print("aborted due to storage!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rootdir', type=str, default='../../../ssd1/dsets/activitynet_captions')
    parser.add_argument('--idfiles', type=list, default=['train_ids.json'])
    parser.add_argument('--videodir', type=str, default='videos')
    parser.add_argument('--savedir', type=str, default='frames')
    parser.add_argument('--shorter', type=int, default=224)
    args = parser.parse_args()
    main(args)
