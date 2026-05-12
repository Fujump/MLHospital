#!/usr/bin/env bash
set -e

reg_alphas=(
  1.4
  1.43
  1.46
  1.49
  1.52
  1.55
  1.58
  1.61
  1.64
  1.67
  1.7
)


for reg_alpha in "${reg_alphas[@]}"; do
  python mlh/examples/train_target_models++.py \
    --mode shadow \
    --epochs 100 \
    --gpu 0 \
    --model resnet18 \
    --dataset CIFAR10 \
    --num_class 10 \
    --inference-dataset CIFAR10 \
    --data-path ../datasets/ \
    --log_path ./trained_models \
    --seed 0 \
    past \
    --reg_weight 1e-3 \
    --reg_alpha "${reg_alpha}" \
    --reg_epoch 50 \
    --reg_norm l1 \
    --reg_clamp 10000
done
