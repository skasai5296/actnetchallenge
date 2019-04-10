# actnetchallenge
repo for activity net challenge 2019

TO-DO:
- [x] complete script for downloading ActivityNet videos
- [ ] complete script for converting .mp4 extensions to .jpg
- [ ] write dataset class for ActivityNet Captions dataset
- [ ] write model for learning from pretrained features of videos

## How to download ActivityNet Dataset
1. download json file from [here](http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)
1. modify `download.sh` and fix the command line argument for root directory
1. `./download.sh`
