#!/bin/bash

# Make data folder relative to script location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../data/

wget "https://mysnu-my.sharepoint.com/:u:/g/personal/aylee2020_seoul_ac_kr/Eflw-aVmqCxAqO2tron2kBgB6TVKPuz2Q_sHwM5DoBqYRQ?e=uCXdpY&download=1" -O $SCRIPT_DIR/../data/ip2p_metrics.zip
wget "https://mysnu-my.sharepoint.com/:u:/g/personal/aylee2020_seoul_ac_kr/ERBqJM-iEshAuKdY4-5joD0Bft20Y8ZzIpPgWNdnjNx23w?e=OIFpw6&download=1" -O $SCRIPT_DIR/../data/metrics.zip

unzip $SCRIPT_DIR/../data/\*.zip -d $SCRIPT_DIR/../data/$1/