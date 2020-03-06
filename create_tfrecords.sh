#!/bin/bash

python create_tfrecords.py \
    --image_dir=./datasets/lfw/val/images \
    --annotations_dir=./datasets/lfw/val/annotations \
    --output=./datasets/lfw/val/shards \
    --num_shards=1324

#python create_tfrecords.py \
#    --image_dir=/disk020/usrs/hagio/FaceBoxes-tensorflow/dataset/val/images/ \
#    --annotations_dir=/disk020/usrs/hagio/FaceBoxes-tensorflow/dataset/val/annotations/ \
#    --output=/disk020/usrs/hagio/face-detection-tensorflow/dataset/val_shards_1_per_tf/ \
#    --num_shards=70000

