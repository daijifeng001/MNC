#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

URL="https://onedrive.live.com/download?resid=F371D9563727B96F!91967&authkey=!AKjrYZBFAfb6JBQ"

echo "Downloading VGG16.mask.caffemodel model..."

mkdir ../imagenet_models/
wget ${URL} -O VGG16.mask.caffemodel 

mv VGG16.mask.caffemodel ../imagenet_models/

URL="https://onedrive.live.com/download?resid=F371D9563727B96F!91966&authkey=!ABoH69DkSk81FwA"

echo "Downloading VGG16.v2.caffemodel model..."

wget ${URL} -O VGG16.v2.caffemodel 

mv VGG16.v2.caffemodel ../imagenet_models/
