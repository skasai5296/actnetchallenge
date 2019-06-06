import sys, os
import random

from PIL import Image
import functools
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.append(os.pardir)
import transforms.spatial_transforms as spt
import transforms.temporal_transforms as tpt
from langmodels.vocab import tokenize

# video loader.
# loads frame indices and gets the images from each frame
def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

# get default video loader.
def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

# get default image loader. accimage is faster than PIL
def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def pil_loader(path):
    img = Image.open(path)
    return img.convert('RGB')
    """
    with Image.open(path) as img:
        return img.convert('RGB')
    """

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

# returns the index <-> video id mapping
def id_loader(path, framepath):
    """
    Args:
        path : string containing path to json file containing video ids
        framepath : string containing path to video frames
    Returns:
        (id2key, key2id) where:
            id2key : list containing video ids
            key2id : dictionary containing mapping of video ids to indexes
    """
    key2id = {}
    with open(path, "r") as f:
        ids = json.load(f)
    # remove "v_" prefix
    id2key = [id[2:] for id in ids if os.path.exists(os.path.join(framepath, id[2:]))]
    for index, videoid in enumerate(id2key):
        key2id[videoid] = index
    return id2key, key2id

"""
preprocess object from activitynet captions dataset.
concats metadata into data.
data attributes: list of dict. list index is the index of video, dict is info about video.
dict : {
    'video_id'      : str, shows the video id.
    'framerate'     : float, shows the framerate per second
    'num_frames'    : int, total number of frames in the video
    'width'         : int, the width of video in pixels
    'height'        : int, the height of video in pixels
    'regions'       : [[int, int], [int, int], ...], list including start and end frame numbers of actions
    'captions'      : [str, str, ...], list including captions for each action
    'segments'      : int, number of actions
}
"""
def preprocess(predata, metadata, key2idx):
    data = [None] * len(key2idx.keys())
    for obj in metadata:
        tmp = {}
        idx = key2idx[obj['video_id']]
        tmp['video_id'] = obj['video_id']
        tmp['framerate'] = obj['framerate']
        tmp['num_frames'] = obj['num_frames']
        tmp['width'] = obj['width']
        tmp['height'] = obj['height']
        data[idx] = tmp
    for v_id, obj in predata.items():
        try:
            idx = key2idx[v_id[2:]]
        except KeyError:
            continue
        fps = data[idx]['framerate']
        regions_sec = obj['timestamps']
        captions = obj['sentences']
        regs = []
        regcnt = 0
        for region in regions_sec:
            # convert into frame duration
            region = [min(int(ts*fps), data[idx]['num_frames']) for ts in region]
            regs.append(region)
            regcnt += 1
        data[idx]['regions'] = regs
        data[idx]['captions'] = captions
        data[idx]['segments'] = regcnt
    return data

"""
preprocess object from activitynet captions dataset. (only for test)
data attributes: list of dict. list index is the index of video, dict is info about video.
dict : {
    'video_id'      : str, shows the video id.
    'framerate'     : float, shows the framerate per second
    'num_frames'    : int, total number of frames in the video
    'width'         : int, the width of video in pixels
    'height'        : int, the height of video in pixels
    'regions'       : [[int, int], [int, int], ...], list including start and end frame numbers of actions
    'segments'      : int, number of actions
}
"""
def preprocess_test(metadata, key2idx):
    data = [None] * len(key2idx.keys())
    lengths = [120, 240, 360, 480, 600, 720, 840, 1200]
    for obj in metadata:
        tmp = {}
        idx = key2idx[obj['video_id']]
        tmp['video_id'] = obj['video_id']
        tmp['framerate'] = obj['framerate']
        tmp['num_frames'] = obj['num_frames']
        tmp['width'] = obj['width']
        tmp['height'] = obj['height']
        regs = []
        for duration in lengths:
            for i in range(10):
                begin = 0 if obj['num_frames'] <= duration else random.randrange(0, obj['num_frames']-duration)
                reg = [begin, min(obj['num_frames']-1, begin+duration)]
                regs.append(reg)
        tmp['regions'] = regs
        tmp['segments'] = len(regs)
        data[idx] = tmp
    # random generation of regions
    return data

class ActivityNetCaptions(Dataset):
    """
    Args:
        root_path (string): Root directory path.
        mode : 'train', 'val', 'test'
        spatial_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load a video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 metadata,
                 mode,
                 vocab,
                 frame_path='frames',
                 is_adaptively_dilated=False,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        self.dilate = is_adaptively_dilated

        idpath = "{}_ids.json".format(mode)
        # save frame root path
        self.framepath = os.path.join(root_path, frame_path)
        self.idx2key, self.key2idx = id_loader(os.path.join(root_path, idpath), self.framepath)

        try:
            assert mode in ['train', 'val', 'test']
        except AssertionError:
            print("mode in ActivityNetCaptions must be one of ['train', 'val', 'test']", True)
        self.category = 'val_1' if mode == 'val' else mode
        self.annfile = "{}.json".format(self.category) if mode is not 'test' else None

        # load annotation files
        if mode != 'test':
            with open(os.path.join(root_path, self.annfile)) as f:
                self.predata = json.load(f)

        # load metadata files
        with open(os.path.join(root_path, metadata)) as f:
            self.meta = json.load(f)

        if mode in ['test','val']:
            self.data = preprocess_test(self.meta, self.key2idx)
        else:
            self.data = preprocess(self.predata, self.meta, self.key2idx)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.vidnum = len(self.data)
        self.vocab = vocab
        self.sample_duration = sample_duration
        self.mode = mode

    def get_clip_from_dur(self, id, dur):
        startframe, endframe = dur[0], dur[1]
        frame_indices = range(startframe, endframe)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        path = os.path.join(self.framepath, id)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        try:
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        except:
            return None
        return clip

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, caption)
            clip : torch.Tensor of size (C, T, H, W)
            caption : torch.LongTensor of size (*,)
        """

        if self.sample_duration == 0:
            id = self.data[index]['video_id']
            reg = self.data[index]['segments']
            return None, None, id, reg

        # fetch different index for incomplete data
        while True:
            id = self.data[index]['video_id']
            try:
                num_actions = self.data[index]['segments']
            except KeyError:
                index = random.randint(0, self.vidnum-1)
                print("no key 'segments', id={}".format(id), flush=True)
                continue

            num_frames = self.data[index]['num_frames']
            num_actions = self.data[index]['segments']

            # get random action segment number from number of actions
            clipnum = random.randint(0, num_actions-1)
            startframe, endframe = self.data[index]['regions'][clipnum]
            frame_indices = list(range(startframe, endframe+1))

            path = os.path.join(self.framepath, id)

            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            # retry when frame indices returns an empty list, result in infinite loop
            if len(frame_indices) < self.sample_duration:
                index = random.randint(0, self.vidnum-1)
                print("length of frame indices ({}) does not match specified size".format(len(frame_indices)), flush=True)
                continue

            clip = self.loader(path, frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            try:
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            except:
                index = random.randint(0, self.vidnum-1)
                print("stacking failed", flush=True)
                continue

            # retry when clip is not expected size (for some reason)
            try:
                assert clip.size(1) == self.sample_duration
            except AssertionError:
                index = random.randint(0, self.vidnum-1)
                print("duration of clip ({}) does not match specified size".format(clip.size(1)), flush=True)
                continue
            break

        """
        tmp = self.data[index]['segments']

        id = self.data[index]['video_id']
        num_frames = self.data[index]['num_frames']
        num_actions = self.data[index]['segments']

        # get random action segment number from number of actions
        clipnum = random.randint(0, num_actions-1)
        startframe, endframe = self.data[index]['regions'][clipnum]
        frame_indices = list(range(startframe, endframe+1))

        path = os.path.join(self.framepath, id)

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # retry when clip is not expected size (for some reason)
        assert clip.size(1) == self.sample_duration
        """

        if self.mode not in ['test', 'val']:
            caption = self.data[index]['captions'][clipnum]
            caption = torch.tensor(self.vocab.return_idx(caption), dtype=torch.long)
        else:
            caption = None

        id = self.data[index]['video_id']
        fps = self.data[index]['framerate']
        reg = [round(startframe/fps, 2), round(endframe/fps, 2)]

        return clip, caption, id, reg

    def __len__(self):
        return self.vidnum



if __name__ == '__main__':
    sp = spt.Compose([spt.CornerCrop(size=224), spt.ToTensor()])
    tp = tpt.Compose([tpt.TemporalRandomCrop(16), tpt.LoopPadding(16)])
    dset = ActivityNetCaptions('/ssd1/dsets/activitynet_captions', 'videometa_train.json', 'train', 'frames', spatial_transform=sp, temporal_transform=tp)
    print(dset[0][0].size())
    print(dset[0][1])




