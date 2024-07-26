#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../checkpoints

wget "https://drive.usercontent.google.com/download?id=1SWF4VIgjYt91iZGqfk6zZFauyLG-tzYz&export=download&authuser=0&confirm=t" -O $SCRIPT_DIR/../checkpoints/epoch=000032-step=000019999.ckpt
wget "https://drive.usercontent.google.com/download?id=1dAbCi1vfvqBihPh8eTVT_XtiLhzxwI3K&export=download&authuser=0&confirm=t" -O $SCRIPT_DIR/../checkpoints/instruct-pix2pix-00-22000.ckpt
