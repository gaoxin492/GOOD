#!/bin/bash
# OE fine-tuning with GOOD-generated OOD samples (ImageNet-100)
CUDA_VISIBLE_DEVICES=0 python -m detection.train_ood_detector \
    --dataset imagenet100 \
    --data_root ./data \
    --num_classes 100 \
    --load checkpoints/classification/imagenet100_resnet34_224.pt \
    --model_arch resnet34 \
    --ood_data_energy logs_energy_in100 \
    --ood_data_knn logs_knn_in100 \
    --ood_data all \
    --energy_weight 1.0 \
    --epochs 200 \
    --learning_rate 0.0001 \
    --batch_size 80 \
    --save checkpoints/detection
