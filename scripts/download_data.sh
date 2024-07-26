#!/bin/bash

# Make data folder relative to script location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../data/

wget "https://drive.usercontent.google.com/download?id=1x7xRBmFEt1xa75-Lb23kNwCqZN4CVR7Z&export=download&authuser=0&confirm=t" -O $SCRIPT_DIR/../data/img_metrics.zip
wget "https://drive.usercontent.google.com/download?id=1k0ocXeuUWSvvRnANiS2pKPbTjkxcISdf&export=download&authuser=0&confirm=t" -O $SCRIPT_DIR/../data/obj_metrics.zip

unzip $SCRIPT_DIR/../data/\*.zip -d $SCRIPT_DIR/../data/$1/
