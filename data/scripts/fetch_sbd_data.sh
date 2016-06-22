#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

FILE=benchmark.tgz
URL=http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/${FILE}
echo "Downloading SBD data..."

wget $URL -O ${FILE}
 
echo "Unzipping..."

mkdir tmp
tar zxvf ${FILE} -C tmp/ --strip-components=1

echo "move it to target source..."

mv -v tmp/dataset/inst/ tmp/dataset/cls/ tmp/dataset/img/ -t ../VOCdevkitSDS/

rm benchmark.tgz
rm -r tmp
