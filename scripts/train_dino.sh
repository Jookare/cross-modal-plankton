#!/bin/bash

# Only GPU 0 will be visible
export CUDA_VISIBLE_DEVICES=0

PREFIX_uto=../data/UTO/fold
PREFIX_lab=../data/LAB/fold

# Train uto
python3 train_dino.py --dataset ${PREFIX_uto} --modelcard ../configs/train/dino/efficientnet_b0_cnn.yaml


# Train lab
for id in {1..5}
do 
    python3 train_dino.py --dataset ${PREFIX_lab}${id} --modelcard ../configs/train/dino/efficientnet_b0_cnn.yaml
done
