# actnetchallenge
repo for activity net challenge 2019
This repository provides a dense video captioning module for ActivityNet Captions Dataset.

TO-DO:
- [x] complete script for downloading ActivityNet videos
- [x] complete script for converting .mp4 videos to .jpg frames
- [x] write dataset class for ActivityNet Captions dataset
- [x] write baseline model for training
- [x] add optional training
- [ ] add validation and testing code
- [x] add Transformer training
- [ ] add BERT training
- [x] add character level training
- [ ] merge code

## How to download ActivityNet Captions Dataset (ActivityNet Videos + Annotations)
1. Download json file for ActivityNet dataset from [here](http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)
1. Modify `download.sh` and fix the command line argument for root directory.
1. Make sure you have at least 300GB on your storage.
1. `bash download.sh`
1. Download json files for ActivityNet Captions dataset from [here](https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip)
1. Run `python utils/add_fps_into_activitynet_json.py ${video_dir} ${}/train.json`
1. Run `python utils/add_fps_into_activitynet_json.py ${video_dir} ${}/val_1.json`
1. Run `python utils/add_fps_into_activitynet_json.py ${video_dir} ${}/val_2.json`
1. Run `python utils/add_fps_into_activitynet_json.py ${video_dir} ${}/test.json`

## How to convert video files to image files
1. Make sure you have at least 1TB and enough Inodes left on your storage.
1. Run `python utils/mp42jpg.py ${video_dir} ${frame_dir} activitynet --n_jobs=${number_of_workers}`

## Training procedures
1. Run `train.py` with configurations (script is in trainscripts.sh)

## Testing procedures
1. Run `test.py` with configurations (script is in testscripts.sh)

## Samples

### Transformer Captions
---
![Transformer Captions](assets/transformer_sample.png)
