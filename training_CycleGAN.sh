#!/bin/bash
export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29502 src/train_cyclegan_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir output/cyclegan_turbo/512-patches_256-cropping-hflip_trial \
    --dataset_folder /home/ricardo/projects/Image2image_applications/data/img2img/512_patches_noval-turbo \
    --train_img_prep resize_512_randomcrop_256x256_hflip \
    --val_img_prep no_resize \
    --learning_rate 1e-5 \
    --max_train_steps 25000 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --tracker_project_name gparmar_unpaired_h2z_cycle_debug_v2 \
    --enable_xformers_memory_efficient_attention \
    --validation_steps 250 \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 \
    --viz_freq 20 \
    --validation_num_images 50 \
    --checkpointing_steps 500 \
    --mixed_precision no \
    --dataloader_num_workers 4 \
    

# --train_img_prep resize_286_randomcrop_256x256_hflip\
# --allow_tf32 \