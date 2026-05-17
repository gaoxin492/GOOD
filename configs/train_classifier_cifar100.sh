#!/bin/bash
# Train ID classifier on CIFAR-100
CUDA_VISIBLE_DEVICES=0 python -m classification.train_classifier \
    --dataset cifar100 \
    --data_root ./data \
    --num_classes 100 \
    --res 32 \
    --model_arch resnet34 \
    --epochs 500 \
    --learning_rate 0.1 \
    --batch_size 160 \
    --save checkpoints/classification \
    --seed 100
