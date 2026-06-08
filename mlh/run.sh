#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=1
smooth_epses=(
0.1
0.2
0.3
0.4
0.5
0.6
)

for smooth_eps in "${smooth_epses[@]}"; do
  python mlh/examples/train_target_models++.py \
    --mode target \
    --epochs 150 \
    --gpu 0 \
    --model resnet18 \
    --dataset CIFAR10 \
    --num_class 10 \
    --inference-dataset CIFAR10 \
    --data-path ../datasets/ \
    --log_path ./trained_models/LabelSmoothing/epoch150/models/resnet18 \
    --seed 0 \
    --lr 0.01 \
    LabelSmoothing \
    --smooth_eps "${smooth_eps}"
done