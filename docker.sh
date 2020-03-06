#!/bin bash

docker run --runtime=nvidia -it --rm -v /disk018/usrs/hagio:/hagio nh122112/tensorflow:stylegan_4
