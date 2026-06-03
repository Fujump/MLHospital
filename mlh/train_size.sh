#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=2
reg_alphas=(
  1.3
)

sample_sizes=(
  10000
  9500
  9000
  8500
  8000
  7500
  7000
  6500
  6000
)

for reg_alpha in "${reg_alphas[@]}"; do
  for sample_size in "${sample_sizes[@]}"; do
    python mlh/examples/train_size_models.py \
      --mode target \
      --epochs 100 \
      --gpu 0 \
      --model resnet18 \
      --dataset CIFAR10 \
      --num_class 10 \
      --inference-dataset CIFAR10 \
      --data-path ../datasets/ \
      --log_path "./trained_models/trainsize/${sample_size}" \
      --seed 0 \
      --lr 0.01 \
      --sample_size "${sample_size}" \
      past \
      --reg_weight 1e-3 \
      --reg_alpha "${reg_alpha}" \
      --reg_epoch 50 \
      --reg_norm l1 \
      --reg_clamp 10000
  done
done
