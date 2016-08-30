#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

URL="https://ggzlla-bn1306.files.1drv.com/y3mNmLgTY25BYV4p7fXFSsZIeTYLcCws1VKjNJDrF7ny1Hy59LWz-RP2-SggiM0uJcllDYcgNoxMPXem6doj9IpeG7L6-H0q8j2kgbssz7nRFXmQXbXOvDgqHLmsEOGSftgFwSJI4tdgXW_-KvnVukpVRUMBThCxkE3w3OSv8PwBi0/mnc_model.caffemodel.h5?download&psid=1"

echo "Downloading mnc model..."

mkdir ../mnc_model
wget ${URL} -O mnc_model.caffemodel.h5 

mv mnc_model.caffemodel.h5 ../mnc_model/
