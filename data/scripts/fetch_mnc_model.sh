#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

URL="https://onedrive.live.com/redir?resid=571EABC0F8C2A19C!1103&authkey=!ALXduVujs-7r6Ug"

echo "Downloading mnc model..."

mkdir ../mnc_model
wget ${URL} -O mnc_model.caffemodel.h5 

mv mnc_model.caffemodel.h5 ../mnc_model/
