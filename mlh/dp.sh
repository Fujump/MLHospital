#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=5

# EPSILONS=(100 200 300 400)
EPSILONS=(300 400)
MAX_GRAD_NORMS=(0.5 1 2 5 10)
DP_DELTA=(1e-6)

for eps in "${EPSILONS[@]}"; do
  for grad_norm in "${MAX_GRAD_NORMS[@]}"; do
    echo "Running DP-SGD with epsilon=${eps}, delta=${DP_DELTA}, max_grad_norm=${grad_norm}"

    python mlh/examples/train_target_dp.py \
      --mode shadow \
      --epochs 150 \
      --gpu 0 \
      --model resnet18 \
      --dataset CIFAR10 \
      --num_class 10 \
      --inference-dataset CIFAR10 \
      --data-path ../datasets/ \
      --log_path ./trained_models/new_DP/lr1e-2 \
      --seed 0 \
      --lr 0.01 \
      dp \
      --dp_epsilon "${eps}" \
      --dp_delta "${DP_DELTA}" \
      --dp_grad_norm "${grad_norm}"
  done
done
