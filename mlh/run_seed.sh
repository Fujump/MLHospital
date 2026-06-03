#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=6
reg_alphas=(
  1.3
)
seeds=("$@")

if [ ${#seeds[@]} -eq 0 ]; then
  seeds=(1 2 3 4 5 6 7 8 9)
fi

for reg_alpha in "${reg_alphas[@]}"; do
  for seed in "${seeds[@]}"; do
    python mlh/examples/train_target_models++.py \
      --mode shadow \
      --epochs 100 \
      --gpu 0 \
      --model resnet18 \
      --dataset CIFAR10 \
      --num_class 10 \
      --inference-dataset CIFAR10 \
      --data-path ../datasets/ \
      --log_path "./trained_models/seed/seed_${seed}" \
      --seed "${seed}" \
      --lr 0.01 \
      past \
      --reg_weight 1e-3 \
      --reg_alpha "${reg_alpha}" \
      --reg_epoch 50 \
      --reg_norm l1 \
      --reg_clamp 10000
  done
done
