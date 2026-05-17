#!/bin/bash
# Train ID classifier on ImageNet-100
CUDA_VISIBLE_DEVICES=0 python -m classification.train_classifier \
    --dataset imagenet100 \
    --data_root ./data \
    --num_classes 100 \
    --res 224 \
    --model_arch resnet34 \
    --epochs 200 \
    --learning_rate 0.001 \
    --batch_size 160 \
    --save checkpoints/classification \
    --seed 100
