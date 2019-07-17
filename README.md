# actnetchallenge: Task 3 (Dense-Captioning Events in Videos)
Repo for activity net challenge 2019: Task 3 (Dense-Captioning Events in Videos)
This repository provides a dense video captioning module for ActivityNet Captions Dataset.

TO-DO:
- [x] complete script for downloading ActivityNet videos
- [x] complete script for converting .mp4 videos to .jpg frames
- [x] write dataset class for ActivityNet Captions dataset
- [x] write baseline model for training
- [x] add optional training
- [ ] add proposal generation code
- [ ] add testing code
- [ ] add Transformer training
- [ ] add BERT training
- [ ] add character level training

## How to download ActivityNet Captions Dataset (ActivityNet Videos + Annotations)
1. Download json file for ActivityNet dataset from [here](http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)
1. Modify `download.sh` and fix the command line argument for root directory to save the dataset. This path will be denoted `$root_path`.
1. Make sure you have at least 300GB on your storage.
1. Run `bash download.sh` to download .mp4 files.
1. Download json files for ActivityNet Captions dataset from [here](https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip)
1. Extract downloaded files to `$root_path`
1. Run `python utils/add_fps_into_activitynet_json.py -v ${video_dir} -s ${root_path}/train.json -o ${save_path}`
1. Run `python utils/add_fps_into_activitynet_json.py -v ${video_dir} -s ${root_path}/val_1.json -o ${save_path}`
1. Run `python utils/add_fps_into_activitynet_json.py -v ${video_dir} -s ${root_path}/val_2.json -o ${save_path}`

## How to convert video files to image files
1. Make sure you have at least 1TB and enough Inodes left on your storage.
1. Run `python utils/mp42jpg.py ${video_dir} ${root_path}/frames activitynet --n_jobs=${number_of_workers}`

## Training procedures
1. Run `train.py` with configurations (script is in `train/trainscript.sh`)

## Testing procedures
1. Proposal Generation is not implemented yet, so prepare a json file with proposals.
1. Run `test.py` with configurations (script is in `eval/eval.sh`)

## Samples

### Transformer Captions
---
![Transformer Captions](assets/transformer_sample.png)
