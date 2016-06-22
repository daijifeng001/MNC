#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

URL=https://www.dropbox.com/s/dsv6q7p6bzdztd0/VGG16.mask.caffemodel?dl=1

echo "Downloading VGG16.mask.caffemodel model..."

mkdir ../imagenet_models/
wget ${URL} -O VGG16.mask.caffemodel 

mv VGG16.mask.caffemodel ../imagenet_models/

URL=https://www.dropbox.com/s/1kptgg6s30wfsw1/VGG16.v2.caffemodel?dl=1

echo "Downloading VGG16.v2.caffemodel model..."

wget ${URL} -O VGG16.v2.caffemodel 

mv VGG16.v2.caffemodel ../imagenet_models/
