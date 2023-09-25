# On Consistency Training for Language-based Image Editing Interface
### [Paper](TODO) | [Data](TODO)

Implementation of ConsEdit, based on original [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix) repository.

## Usage

Use this model for editing your own images by following the instructions below.

### Set up a conda environment, and download the appropriate checkpoint:
```
conda env create -f environment.yaml
conda activate ip2p
bash scripts/TODO.sh
```

### Edit a single image:
```
python edit_cli.py --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"

# Optionally, you can specify parameters to tune your result:
# python edit_cli.py --steps 100 --resolution 512 --seed 1371 --cfg-text 7.5 --cfg-image 1.2 --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"
```

## How to reproduce results

Our code for calculating metrics are included in the `metrics` folder. We calculated two types of metrics: object-level metrics and image-level metrics. First download test dataset, run ConsEdit to get results, run YOLOv7 to get labels, and lastly run our metrics code. Follow the details below.

### Download test dataset

```
bash scripts/TODO.sh
```

### Run ConsEdit

Get outputs for calculating object-level metrics first.
```
for i in "1.0" "1.2" "1.4" "1.6" "1.8" "2.0" "2.2"; do `python edit_cli.py --input data/metrics/test_images/ --output data/metrics/ip2p_outputs/ --edit data/metrics/test_edits.pickle --ckpt checkpoints/instruct-pix2pix-00-22000.ckpt --cfg-image ${i}`; done

for i in "0.2" "0.6" "1.0" "1.4" "1.8" "2.2" "2.6" "3.0" "3.4" "3.8"; do `python edit_cli.py --input data/metrics/test_images/ --output data/metrics/cons_outputs/ --edit data/metrics/test_edits.pickle --ckpt checkpoints/epoch\=000032-step\=000019999.ckpt --cfg-image ${i}`; done
```

Then get outputs for calculating image-level metrics.
```
for i in "1.0" "1.2" "1.4" "1.6" "1.8" "2.0" "2.2"; do `python edit_cli.py --input data/ip2p_metrics/test_images/ --output data/ip2p_metrics/ip2p_outputs/ --edit data/ip2p_metrics/test_edits.pickle --ckpt checkpoints/instruct-pix2pix-00-22000.ckpt --cfg-image ${i}`; done

for i in "1.0" "1.2" "1.4" "1.6" "1.8" "2.0" "2.2"; do `python edit_cli.py --input data/ip2p_metrics/test_images/ --output data/ip2p_metrics/cons_outputs/ --edit data/ip2p_metrics/test_edits.pickle --ckpt checkpoints/epoch\=000032-step\=000019999.ckpt --cfg-image ${i}`; done
```

Note: This would take a long time. If more gpus are available, try to remove the for loop and process in parallel.

### Run YOLOv7

Setup YOLOv7 first.
```
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7

conda create --name yolo python=3.8
pip install -r requirements.txt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

Then run YOLOv7 for all of the scales
```
bash TODO.sh
```

### Calculate & plot metrics

```
for i in {0..6}; do `python metrics/im_metrics.py --gpu ${i} 2>&1 &`; done # should have at least 7 gpus
python metrics/obj_metrics.py
python plot_metrics.py
```

## Acknowledgements

- https://github.com/timothybrooks/instruct-pix2pix
- https://github.com/WongKinYiu/yolov7

## BibTeX

```
TODO
```
