#!/bin/bash
# Evaluate OOD detection on ImageNet-100
CUDA_VISIBLE_DEVICES=0 python -m detection.evaluate \
    --dataset imagenet100 \
    --data_root ./data \
    --num_classes 100 \
    --model_arch resnet34 \
    --checkpoint checkpoints/detection/imagenet100_resnet34_224_last_model.pt \
    --ood_test_root ./data/test_data_all \
    --cache_dir ./cache \
    --K 100 \
    --a 0.5 \
    --batch_size 256 \
    --output results_in100.txt
