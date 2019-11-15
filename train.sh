#!/bin bash

# source model training
#python3 main.py --phase pretrain --sub_dir pretrain_face --gpu_num=8

## both models training
#python3 main.py --phase train --sub_dir sim2city_3 --gpu_num=4

## test
#python3 main.py --phase test --sub_dir pretrain_face --source_only_test True --MT_test True

python3 main.py --phase test --sub_dir pretrain_face --source_only_test True --gpu_num=4
