import sys, os
import shutil
import cv2
import json

import argparse

# video_path    : where the video exists
# tar_path      : where the frames will be saved
# filename      : name to append in front of frames

def vid2frames(video_path, tar_path):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    loglist = []

    # Make the target directory if it doesn't exist.
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    # for files in video directory,
    for i, filename in enumerate(os.listdir(video_path)):
        info = {}
        name, ext = os.path.splitext(filename)
        if ext is '.mp4':

            # get framerate of video
            if int(major_ver) < 3:
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)

            num_frames = 0
            cap = cv2.VideoCapture(os.path.join(video_path, filename))
            # read frame and save until capture is done
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break
                # frame number will be filled with zeros ex) 1 -> 000001
                tar = os.path.join(tar_path, 'vid_{:06d}_frame_{:06d}.jpg'.format(i, num_frames))
                cv2.imwrite(tar, frame)
                num_frames += 1

            cap.release()

            info['num_frames'] = num_frames
            info['video_name'] = filename
            info['video_idx'] = i
            info['framerate'] = fps

            print("Done with converting {} to frames, saved in {}.".format(os.path.join(video_path, filename), tar_path))
            print("{} frames in total, {} fps".format(i, fps), flush=True)

        loglist.append(info)

    return loglist

def main(args):
    if os.path.exists(args.logfile):
        sys.exit("log already exists")
    logs = vid2frames(args.videodir, args.savedir)
    with open(args.logfile, 'w+') as f:
        json.dump(logs, f)
    print("done!! Metadata in {}".format(args.logfile))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videodir', type=str, default='../../../ssd1/dsets/activitynet_captions/videos')
    parser.add_argument('--savedir', type=str, default='../../../ssd1/dsets/activitynet_captions/frames')
    parser.add_argument('--logfile', type=str, default='../../../ssd1/dsets/activitynet_captions/videometa.json')
    args = parser.parse_args()
    main(args)
