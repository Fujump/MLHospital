#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=2
reg_alphas=(
  1.3
  1.4
  1.5
  1.6
)

for reg_alpha in "${reg_alphas[@]}"; do
  python mlh/examples/train_target_models++.py \
    --mode shadow \
    --epochs 100 \
    --gpu 0 \
    --model dense121 \
    --dataset CIFAR10 \
    --num_class 10 \
    --inference-dataset CIFAR10 \
    --data-path ../datasets/ \
    --log_path ./trained_models/models/dense121 \
    --seed 0 \
    --lr 0.01 \
    past \
    --reg_weight 1e-3 \
    --reg_alpha "${reg_alpha}" \
    --reg_epoch 50 \
    --reg_norm l1 \
    --reg_clamp 10000
done