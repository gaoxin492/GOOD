#!/bin/bash
# OE fine-tuning with GOOD-generated OOD samples (CIFAR-100)
CUDA_VISIBLE_DEVICES=0 python -m detection.train_ood_detector \
    --dataset cifar100 \
    --data_root ./data \
    --num_classes 100 \
    --load checkpoints/classification/cifar100_resnet34_32.pt \
    --model_arch resnet34 \
    --ood_data_energy logs_energy_cifar100 \
    --ood_data_knn logs_knn_cifar100 \
    --ood_data all \
    --energy_weight 1.0 \
    --epochs 200 \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --save checkpoints/detection
