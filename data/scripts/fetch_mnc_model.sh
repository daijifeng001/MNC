#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

URL=https://www.dropbox.com/s/5fvbdt30i67bo24/mnc_model.caffemodel.h5?dl=1

echo "Downloading mnc model..."

mkdir ../mnc_model
wget ${URL} -O mnc_model.caffemodel.h5 

mv mnc_model.caffemodel.h5 ../mnc_model/
