#!/bin/bash
git submodule init
git submodule update

wget -c http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar xvzf vgg_16_2016_08_28.tar.gz

cd data
./download.sh

