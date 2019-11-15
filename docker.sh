#!/bin bash

docker run --runtime=nvidia -it --rm -p 6060:6060 -v /disk011/usrs/hagio:/hagio nh122112/tensorflow:1.12.1
#docker run --runtime=nvidia -it --rm -p 6060:6060 -v /disk011/usrs/hagio:/hagio nh122112/tensorflow:1.13.2.00
