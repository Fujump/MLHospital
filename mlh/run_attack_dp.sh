#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PYTHON="${PYTHON:-/data/home/zhanghx/.conda/envs/PAST/bin/python}"

model="${MODEL:-resnet18}"
dataset="${DATASET:-CIFAR10}"
training_name="${TRAINING_NAME:-Normal_L1_eps300.0_delta1e-05_clip10.0}"
model_dir="${MODEL_DIR:-/data/home/zhanghx/huq/MLHospital/trained_models/new_DP/lr1e-2/${dataset}/${training_name}}"
checkpoint_file="${CHECKPOINT_FILE:-${model}_model.pth}"
seed="${SEED:-0}"
attack_type="${ATTACK_TYPE:-metric-based}"
result_group="${RESULT_GROUP:-new_DP/lr1e-2}"
result_root="${RESULT_ROOT:-./attack_results/${result_group}}"

target_path="${model_dir}/target/${checkpoint_file}"
shadow_path="${model_dir}/shadow/${checkpoint_file}"
attack_log_path="${result_root}/${model}/${dataset}/${training_name}/Defense/"

echo "[${model} ${training_name}] target=${target_path}"
echo "[${model} ${training_name}] shadow=${shadow_path}"

mkdir -p "${attack_log_path}"

"${PYTHON}" mlh/examples/mia_example.py \
  --gpu 0 \
  --model "${model}" \
  --dataset "${dataset}" \
  --num-class 10 \
  --attack_log_path "${attack_log_path}" \
  --attack_type "${attack_type}" \
  --batch-size 128 \
  --target_path "${target_path}" \
  --shadow_path "${shadow_path}" \
  --seed "${seed}" \
  2>&1 | tee "${attack_log_path}/${attack_type}.log"
