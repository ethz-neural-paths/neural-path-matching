#!/bin/bash
git submodule init
git submodule update

cd data
./download.sh

