#!/bin/bash

# copy darknet weight files, continue '-c' if partially downloaded
wget -c https://pjreddie.com/media/files/yolov3.weights
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
wget -c https://pjreddie.com/media/files/yolov3-spp.weights

# yolov3 pytorch weights
# download from Google Drive: https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI

# darknet53 weights (first 75 layers only)
wget -c https://pjreddie.com/media/files/darknet53.conv.74

