#!/bin/bash

# Make data folder relative to script location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../data/

wget "https://mysnu-my.sharepoint.com/:u:/g/personal/aylee2020_seoul_ac_kr/ER4VmkwYok1NujrktVo6F38BWXRROu6-1AKr2bODJqnCLA?e=6feTbQ&download=1" -O $SCRIPT_DIR/../data/img_metrics.zip
wget "https://mysnu-my.sharepoint.com/:u:/g/personal/aylee2020_seoul_ac_kr/EUf-B6Np2JhEvyBjdue4QgQBxUTGlLgLz4qgnKmI99az3g?e=KxwYD7&download=1" -O $SCRIPT_DIR/../data/obj_metrics.zip

unzip $SCRIPT_DIR/../data/\*.zip -d $SCRIPT_DIR/../data/$1/