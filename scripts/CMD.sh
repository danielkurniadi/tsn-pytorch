#! /bin/bash

## HMDB51_BNI_ARP_seg7
CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py hmdb51 RGB data/hmdb51_splits/hmdb51_train_split_0.txt data/hmdb51_splits/hmdb51_val_split_0.txt \
--arch BNInception --num_segments 7 --ext .jpg -b 48 --img_prefix arp --print-freq 5 --eval-freq 1 --epochs 45

## 
CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py hmdb51 RGB data/hmdb51_splits/hmdb51_train_split_0.txt data/hmdb51_splits/hmdb51_val_split_0.txt \
--arch BNInception --num_segments 3 --ext .jpg -b 48 --img_prefix arp --print-freq 5 --eval-freq 1 --epochs 45

##
CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py hmdb51 RGB data/hmdb51_splits/hmdb51_train_split_0.txt data/hmdb51_splits/hmdb51_val_split_0.txt \
--arch BNInception --num_segments 7 --ext .jpg -b 48 --img_prefix img --print-freq 5 --eval-freq 1 --epochs 45

##
CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py hmdb51 Flow data/hmdb51_flow_splits/hmdb51_flow_train_split_0.txt data/hmdb51_flow_splits/hmdb51_flow_val_split_0.txt \
--arch BNInception --num_segments 3 --ext .jpg -b 48 --flow_prefix flow --print-freq 5 --eval-freq 1 --epochs 45

##
CUDA_VISIBLE_DEVICES=1,2,3 python3 main.py saag01 Flow ./data/saa_aggression_splits/saag01_arp_train_split_0.txt ./data/saa_aggression_splits/saag01_arp_val_split_0.txt  \
--arch BNInception --num_segments 3 --ext .jpg -b 48 --flow_prefix flow --lr 0.001 --lr_steps 20 40 --gd 20 --epochs 80 --eval-freq 1 --print-freq 5  

## TEST
CUDA_VISIBLE_DEVICES=1,2 python3 test.py saag01 RGB data/roselab_splits/arp_rgb_roselab_test_split.txt \
checkpoints/SAAG01_BNI_ARP_7_model_best.pth.tar --arch BNInception --img_prefix arp --ext .png -b 5 --print_freq 1 --num_segments 7

## TEST
CUDA_VISIBLE_DEVICES=1,2 python3 test.py saag01 Flow data/saa_aggression_splits/saag01_arp_test_split.txt \
checkpoints/SAAG01_BNI_Flow_3_model_best.pth.tar --arch BNInception --flow_prefix flow --ext .jpg -b 5 --print_freq 1 --num_segments 3
