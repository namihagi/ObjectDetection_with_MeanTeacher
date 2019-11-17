#!/bin bash

# source model training
#python3 main.py --phase pretrain --sub_dir pretrain_face --gpu_num=8

## both models training
#python3 main.py --phase train --sub_dir sim2city_3 --gpu_num=4

## test
#python3 main.py --phase test --sub_dir pretrain_face --source_only_test True --MT_test True

## gpu04
#python3 main.py \
#          --phase pretrain \
#          --gpu_num=4 \
#          --sub_dir pretrain_face_gpu_4 \
#          --pretrain_ckpt_sub_dir pretrain_face_gpu_4 \
#          --source_dir face \
#          --source_dir ball

# gpu11
python3 main.py \
          --phase pretrain \
          --gpu_num=1 \
          --sub_dir pretrain_face_gpu_1 \
          --pretrain_ckpt_sub_dir pretrain_face_gpu_1 \
          --source_dir face \
          --source_dir ball

## gpu15
#python3 main.py \
#          --phase pretrain \
#          --gpu_num=8 \
#          --sub_dir pretrain_face_gpu_8 \
#          --pretrain_ckpt_sub_dir pretrain_face_gpu_8 \
#          --source_dir face \
#          --source_dir ball




