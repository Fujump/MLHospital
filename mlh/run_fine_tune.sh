#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=1
reg_alphas=(
  1.4
  1.45
  1.5
  1.55
  1.6
  1.65
  1.7
)
reg_weights=(
  0.001
  0.0007
  0.0005
  0.0003
)
reg_lrs=(
  0.02
  0.01
  0.007
  0.005
  0.003
)

for reg_alpha in "${reg_alphas[@]}"; do
  for reg_weight in "${reg_weights[@]}"; do
    for reg_lr in "${reg_lrs[@]}"; do
      python mlh/examples/past_target_models.py \
        --mode target \
        --epochs 100 \
        --gpu 0 \
        --model resnet18 \
        --dataset CIFAR10 \
        --num_class 10 \
        --inference-dataset CIFAR10 \
        --data-path ../datasets/ \
        --log_path ./trained_models/fine_tune \
        --seed 0 \
        --lr 0.01 \
        past \
        --reg_weight "${reg_weight}" \
        --reg_alpha "${reg_alpha}" \
        --reg_epoch 50 \
        --reg_norm l1 \
        --reg_clamp 10000 \
        --reg_lr "${reg_lr}"
    done
  done
done
