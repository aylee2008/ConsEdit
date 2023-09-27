#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../checkpoints

wget "https://mysnu-my.sharepoint.com/:u:/g/personal/aylee2020_seoul_ac_kr/ETo38lq2I3BBpJ_YkKuR5JUBiqBqCUHd5xLrjADbbqh10w?e=3CxVt1&download=1" -O $SCRIPT_DIR/../checkpoints/epoch=000032-step=000019999.ckpt
#wget "https://mysnu-my.sharepoint.com/:u:/g/personal/aylee2020_seoul_ac_kr/EXUsCsfbbwtIifwfc7MgYsYB8KxYd1XDJlDGUQoZ-6fRLQ?e=QZyseE&download=1" -O $SCRIPT_DIR/../checkpoints/instruct-pix2pix-00-22000.ckpt
