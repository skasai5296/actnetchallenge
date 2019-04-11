import sys, os
import time

from PIL import Image
import functools
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

# 
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
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def id_loader(path):
    """
    Args:
        path : string containing path to json file containing video ids
    Returns:
        (id2key, key2id) where:
            id2key : list containing video ids
            key2id : dictionary containing mapping of video ids to indexes
    """
    key2id = {}
    with open(path, "r") as f:
        id2key = json.load(f)
    for index, videoid in enumerate(id2key):
        key2id[videoid] = index
    print(key2id)
    return id2key, key2id

# preprocess object from activitynet captions dataset.
def preprocess(predata):
    pass

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
                 mode,
                 is_adaptively_dilated=False,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        self.dilate = is_adaptively_dilated

        idpath = "{}_ids.json".format(mode)
        self.category = 'val_1' if mode is 'val' else mode
        self.annfile = "{}.json".format(self.category) if mode is not 'test' else None

        self.id2key, self.key2id = id_loader(os.path.join(root_path, idpath))

        if self.annfile is not None:
            with open(os.path.join(root_path, self.annfile)) as f:
                self.predata = json.load(f)

        self.data = preprocess(self.predata)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, segments) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dset = ActivityNetCaptions('../../../ssd1/dsets/activitynet_captions', 'train')
