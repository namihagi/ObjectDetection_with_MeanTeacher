#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main_input_feature.py \
  --type "cyclegan" \
  --batch_size=1 \
  --ckpt_dir="./checkpoint/pretrain_face_batch_norm" \
  --input_npy_dir="../CycleGAN-with-multiple-gpu/result/ball-disk020-to-lfw-features/train/A2B" \
  --input_image_dir="../CycleGAN-with-multiple-gpu/datasets/ball-web-disk020/set_base/train/images" \
  --output_dir="../CycleGAN-with-multiple-gpu/result/ball-disk020-to-lfw-features/train/A2B_prediction"

CUDA_VISIBLE_DEVICES=0 python3 main_input_feature.py \
  --type "cyclegan" \
  --batch_size=1 \
  --ckpt_dir="./checkpoint/pretrain_face_batch_norm" \
  --input_npy_dir="../CycleGAN-with-multiple-gpu/result/ball-disk020-to-lfw-features/test/A2B" \
  --input_image_dir="../CycleGAN-with-multiple-gpu/datasets/ball-web-disk020/set_base/test/images" \
  --output_dir="../CycleGAN-with-multiple-gpu/result/ball-disk020-to-lfw-features/test/A2B_prediction"

#StyleganDir="00023-sgan-cycle-aeroplane-only2ffhq-512-8gpu"
#CUDA_VISIBLE_DEVICES=2 python3 main_input_feature.py \
#          --type "image" \
#          --batch_size=1 \
#          --ckpt_dir="./checkpoint/pretrain_face_gpu_1" \
#          --input_npy_dir="/hagio/stylegan/results/$StyleganDir/feature_npy" \
#          --input_image_dir="/hagio/datasets/voc2007/each_class_set/train/aeroplane/cliped_images" \
#          --output_dir="./box_result/$StyleganDir"

#--input_image_dir="/hagio/datasets/voc2007/each_class_set/train/aeroplane/images"
#--input_image_dir="/hagio/datasets/ball/cliped_images"

#python3 main_input_feature.py \
#          --batch_size=1 \
#          --ckpt_dir="./checkpoint/pretrain_face_gpu_1" \
#          --input_npy_dir="/hagio/stylegan/results/$StyleganDir/feature_npy" \
#          --input_image_dir="/hagio/datasets/ball/extracted_images" \
#          --output_dir="./box_result/$StyleganDir"
