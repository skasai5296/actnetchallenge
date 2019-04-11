# actnetchallenge
repo for activity net challenge 2019

TO-DO:
- [x] complete script for downloading ActivityNet videos
- [ ] complete script for converting .mp4 videos to .jpg frames
- [ ] write dataset class for ActivityNet Captions dataset
- [ ] write model for learning from pretrained features of videos
- [ ] merge code

## How to download ActivityNet Captions Dataset
1. Download json file for ActivityNet dataset from [here](http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)
1. Modify `download.sh` and fix the command line argument for root directory.
1. `bash download.sh`
1. Download json files for ActivityNet Captions dataset from [here](https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip)
