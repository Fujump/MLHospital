#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
PYTHON="/data/home/zhanghx/.conda/envs/PAST/bin/python"

model_dir="${1:-trained_models/RelaxLoss/epoch150/models/resnet18/CIFAR10/RelaxLoss_0.6}"
model_name="$(basename "${model_dir}")"
seed=0
attack_type="metric-based"

target_path="${model_dir}/target/resnet18_model.pth"
shadow_path="${model_dir}/shadow/resnet18_model.pth"
attack_log_path="./attack_results/RelaxLoss/epoch150/${model_name}/Defense"

mkdir -p "${attack_log_path}"

"${PYTHON}" mlh/examples/mia_example.py \
  --gpu 0 \
  --model resnet18 \
  --dataset CIFAR10 \
  --num-class 10 \
  --attack_log_path "${attack_log_path}" \
  --attack_type "${attack_type}" \
  --batch-size 128 \
  --target_path "${target_path}" \
  --shadow_path "${shadow_path}" \
  --seed "${seed}" \
  2>&1 | tee "${attack_log_path}/${attack_type}.log"
