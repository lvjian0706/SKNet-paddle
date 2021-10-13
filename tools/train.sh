#!/usr/bin/env bash

# for single card train
# python3.7 tools/train.py -c ppcls/configs/ImageNet/SKNet/SKNet50.yaml

# for multi-cards train
python3.7 -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
    -c ppcls/configs/ImageNet/SKNet/SKNet50.yaml