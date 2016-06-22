#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

FILE=MCG-Pascal-Main_trainvaltest_2012-proposals.tgz
URL=https://data.vision.ee.ethz.ch/jpont/mcg/${FILE}
echo "Downloading MCG proposals data..."

wget $URL -O ${FILE}
 
echo "Unzipping..."

mkdir tmp
tar zxvf ${FILE} -C tmp/ --strip-components=1

echo "move it to target source..."

mkdir ../MCG-raw/

mv tmp/* ../MCG-raw/

rm ${FILE}
rm -r tmp
