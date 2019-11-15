#!/bin/bash

python create_tfrecords.py \
    --image_dir=/disk020/usrs/hagio/FaceBoxes-tensorflow/dataset/train/images/ \
    --annotations_dir=/disk020/usrs/hagio/FaceBoxes-tensorflow/dataset/train/annotations/ \
    --output=/disk020/usrs/hagio/face-detection-tensorflow/dataset/train_shards_1_per_tf/ \
    --num_shards=16106

python create_tfrecords.py \
    --image_dir=/disk020/usrs/hagio/FaceBoxes-tensorflow/dataset/val/images/ \
    --annotations_dir=/disk020/usrs/hagio/FaceBoxes-tensorflow/dataset/val/annotations/ \
    --output=/disk020/usrs/hagio/face-detection-tensorflow/dataset/val_shards_1_per_tf/ \
    --num_shards=2845

