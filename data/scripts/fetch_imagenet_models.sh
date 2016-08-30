#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

URL="https://ggzola-bn1306.files.1drv.com/y3mmaGVdHr-9xkT7byW7D9sUXx3DK9JPsql80j3VgMQJ5eYtonjZaS3F2CIbaQNyM36NYeH_J8B0mxFSpaaXl-2zXxEb1K4gETBBgQYomF9k1K4R61PaeaYrqtPzpAoTPUSp-sgseEK0F0iam_icLOcrlacYIV_5gK7UwycmP87U4A/VGG16.mask.caffemodel?download&psid=1"

echo "Downloading VGG16.mask.caffemodel model..."

mkdir ../imagenet_models/
wget ${URL} -O VGG16.mask.caffemodel 

mv VGG16.mask.caffemodel ../imagenet_models/

URL="https://ggznla-bn1306.files.1drv.com/y3m-3LjmEpwi5OgbGunwG_K2JsCa4hUMReNdLfmB3IxmMDTtmDiRY2X_J_qyIKMnmHq56EBmjrqxUs4ud7OU08XqxiptxnDAMIMwK1_pEZJ_3QPOUizoP3dQa0ljR3lZ20_y0owu4jWwgvkU9EprK4BSJ4YXTLGtM1jtEBNoYdQZzg/VGG16.v2.caffemodel?download&psid=1"

echo "Downloading VGG16.v2.caffemodel model..."

wget ${URL} -O VGG16.v2.caffemodel 

mv VGG16.v2.caffemodel ../imagenet_models/
