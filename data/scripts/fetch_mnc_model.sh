#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

URL="https://onedrive.live.com/download?resid=F371D9563727B96F!91968&authkey=!AGcNz7xSH5E98zg"

echo "Downloading mnc model..."

mkdir ../mnc_model
wget ${URL} -O mnc_model.caffemodel.h5 

mv mnc_model.caffemodel.h5 ../mnc_model/
