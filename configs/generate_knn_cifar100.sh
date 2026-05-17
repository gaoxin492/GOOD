#!/bin/bash
# Generate KNN-guided OOD samples for CIFAR-100
CUDA_VISIBLE_DEVICES=0 python -m generation.generate \
    --dataset cifar100 \
    --data_type text2image \
    --task image_knn \
    --target OOD \
    --guide_network resnet34 \
    --num_classes 100 \
    --load checkpoints/classification/cifar100_resnet34_32.pt \
    --cache_dir ./cache \
    --K 100 \
    --model_name_or_path stable-diffusion-v1-5 \
    --guidance_name tfg \
    --image_size 512 \
    --inference_steps 100 \
    --eta 1.0 \
    --clip_x0 False \
    --rho 1.0 \
    --mu 1.0 \
    --sigma 0.001 \
    --eps_bsz 1 \
    --per_sample_batch_size 5 \
    --num_samples 100 \
    --logging_dir logs_knn_cifar100
