#!/bin/bash

## source model training
#CUDA_VISIBLE_DEVICES=0 python3 main.py \
#  --phase pretrain \
#  --sub_dir pretrain_face_batch_norm \
#  --pretrain_ckpt_sub_dir pretrain_face_batch_norm \
#  --gpu_num 1 \
#  --batch_size_per_gpu 16

## both models training
#python3 main.py --phase train --sub_dir sim2city_3 --gpu_num=4

# FDDB val
python3 main.py \
  --phase val \
  --source_only_test True \
  --sub_dir mange_image \
  --gpu_num 1 \
  --batch_size_per_gpu 1 \
  --pretrain_ckpt_sub_dir pretrain_face_batch_norm \
  --result_dir FDDB_val \
  --source_dir face \
  --target_dir ball

## test
#python3 main.py \
#  --phase test \
#  --source_only_test True \
#  --sub_dir mange_image \
#  --gpu_num 1 \
#  --batch_size_per_gpu 1 \
#  --pretrain_ckpt_sub_dir pretrain_face_batch_norm \
#  --result_dir ./result/lfw \
#  --source_dir lfw \
#  --target_dir ball

## get feature maps
#python3 main.py \
#          --phase get_feature_maps \
#          --gpu_num 1 \
#          --batch_size_per_gpu 1 \
#          --sub_dir test \
#          --pretrain_ckpt_sub_dir pretrain_face_batch_norm \
#          --result_dir ./result/lfw \
#          --source_dir lfw \
#          --target_dir ball
