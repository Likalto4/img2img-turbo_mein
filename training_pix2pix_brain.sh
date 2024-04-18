#!/bin/bash
accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir output/pix2pix_turbo/vanilla_T1-T2 \
    --dataset_folder /home/ricardo/projects/Image2image_applications/data/img2img/brain/normalized_rgb \
    --resolution 256 \
    --train_batch_size 1 \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --report_to "wandb" --tracker_project_name "pix2pix_turbo_brain"