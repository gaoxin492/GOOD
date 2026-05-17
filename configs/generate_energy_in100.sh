#!/bin/bash
# Generate energy-guided OOD samples for ImageNet-100
CUDA_VISIBLE_DEVICES=0 python -m generation.generate \
    --dataset imagenet100 \
    --data_type text2image \
    --task image_energy \
    --target OOD \
    --guide_network resnet34 \
    --num_classes 100 \
    --load checkpoints/classification/imagenet100_resnet34_224.pt \
    --model_name_or_path stable-diffusion-v1-5 \
    --guidance_name tfg \
    --image_size 512 \
    --inference_steps 100 \
    --eta 1.0 \
    --clip_x0 False \
    --rho 0.5 \
    --mu 0.5 \
    --sigma 0.1 \
    --eps_bsz 1 \
    --per_sample_batch_size 4 \
    --num_samples 100 \
    --logging_dir logs_energy_in100
