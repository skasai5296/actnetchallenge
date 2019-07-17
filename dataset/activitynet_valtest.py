import sys, os
sys.path.append(os.pardir)

from PIL import Image
import functools
import json
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import transforms.spatial_transforms as spt
import transforms.temporal_transforms as tpt

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


class ActivityNetCaptions_Val(Dataset):
    """
    Args:
        root_path (string): Root directory path.
        n_samples_for_each_video (int): Number of actions to retrieve per video.
        spatial_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load a video given its path and frame indices.
    """
    def __init__(self,
                 root_path,
                 frame_path='frames',
                 ann_path=['val_1_fps.json', 'val_2_fps.json'],
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        # save frame root path
        self.frm_path = os.path.join(root_path, frame_path)

        # load annotation files
        self.data = []
        for path in ann_path:
            with open(os.path.join(root_path, path)) as f:
                ann = json.load(f)

            for id, obj in ann.items():
                if "fps" not in obj.keys():
                    continue
                content = {}
                content["id"] = id
                lastfile = os.listdir(os.path.join(self.frm_path, id))[-1]
                lastframenum = int(re.findall(r'\d+', lastfile)[0])
                content["duration"] = lastframenum
                content["duration"] = obj["duration"]
                content["sentences"] = obj["sentences"]
                content["timestamps"] = obj["timestamps"]
                content["fps"] = obj["fps"]
                self.data.append(content)


        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.len = len(self.data)
        print("Validation dataset length: ", self.len)
        self.sample_duration = sample_duration
        self.n_actions = n_samples_for_each_video

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict containing the following:
            'id': string
            'duration': int, length of clip in seconds
            'sentences': list of strings, caption
            'timestamps': list of [int, int], shows the beginning and end frames of action
            'fps': float, framerate of video, some videos don't contain this
            'clip' : list of torch.Tensor of size (C, T, H, W)
        """
        id = self.data[index]['id']
        duration = self.data[index]['duration']
        sentences = []
        timestamps = []
        fps = self.data[index]['fps']
        fidlist = []

        for num, (sentence, timestamp) in enumerate(zip(self.data[index]['sentences'], self.data[index]['timestamps'])):
            if num == self.n_actions:
                break
            sentences.append(sentence)
            begin_frame = min(int(fps * duration), max(1, int(fps * timestamp[0])))
            end_frame = min(duration, int(fps * timestamp[1]))
            timestamps.append([begin_frame, end_frame])
            frame_indices = list(range(begin_frame, end_frame))
            fidlist.append(frame_indices)

        frame_indices = fidlist[np.random.randint(0, len(sentences))]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.loader(os.path.join(self.frm_path, id), frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        try:
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            assert clip.size(1) == self.sample_duration
        except:
            print("stack failed or clip is not right size", flush=True)
            print(len(clip), flush=True)
            print([cl.size() for cl in clip], flush=True)
            print(id)

        return {'id': id, 'duration': duration, 'sentences': sentences, 'timestamps': timestamps, 'fps': fps, 'clip': clip}

    def __len__(self):
        return self.len

class ActivityNetCaptions_Test(Dataset):
    """
    Args:
        root_path (string): Root directory path.
        n_samples_for_each_video (int): Number of actions to retrieve per video.
        spatial_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load a video given its path and frame indices.
    """
    def __init__(self,
                 root_path,
                 frame_path='frames',
                 ann_path='test_fps.json',
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        # save frame root path
        self.frm_path = os.path.join(root_path, frame_path)

        # load annotation files
        self.data = []
        with open(os.path.join(root_path, path)) as f:
            ids = json.load(f)

        for id, obj in ids.items():
            if "fps" not in obj.keys():
                continue
            content = {}
            content["id"] = id
            content["duration"] = obj["duration"]
            content["timestamps"] = obj["timestamps"]
            content["fps"] = obj["fps"]
            self.data.append(content)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.len = len(self.data)
        print("Validation dataset length: ", self.len)
        self.sample_duration = sample_duration
        self.n_actions = n_samples_for_each_video

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict containing the following:
            'id': string
            'duration': int, length of clip in seconds
            'timestamps': list of [int, int], shows the beginning and end frames of action
            'fps': float, framerate of video, some videos don't contain this
            'clip' : list of torch.Tensor of size (C, T, H, W)
        """
        id = self.data[index]['id']
        duration = self.data[index]['duration']
        timestamps = []
        fps = self.data[index]['fps']
        fidlist = []

        for num, timestamp in enumerate(self.data[index]['timestamps']):
            if num == self.n_actions:
                break
            begin_frame = max(1, int(fps * timestamp[0]))
            end_frame = min(int(fps * duration-3), int(fps * timestamp[1]))
            timestamps.append([begin_frame, end_frame])
            frame_indices = list(range(begin_frame, end_frame))
            fidlist.append(frame_indices)

        clips = []
        for frame_indices in fidlist:
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            clip = self.loader(os.path.join(self.frm_path, id), frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            try:
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
                assert clip.size(1) == self.sample_duration
            except:
                print("stack failed or clip is not right size", flush=True)
                print(len(clip), flush=True)
                print([cl.size() for cl in clip], flush=True)
                print(id, flush=True)
            clips.append(clip)

        return {'id': id, 'duration': duration, 'timestamps': timestamps, 'fps': fps, 'clips': clips}

    def __len__(self):
        return self.len


if __name__ == '__main__':
    sp = spt.Compose([spt.CornerCrop(size=224), spt.ToTensor()])
    tp = tpt.Compose([tpt.TemporalRandomCrop(16), tpt.LoopPadding(16)])
    dset = ActivityNetCaptions_Train('/ssd1/dsets/activitynet_captions', spatial_transform=sp, temporal_transform=tp)
    print(dset[0][0].size())
    print(dset[0][1])




