#!/bin/bash

wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
unzip data_scene_flow.zip

wget -c http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar xvzf vgg_16_2016_08_28.tar.gz

