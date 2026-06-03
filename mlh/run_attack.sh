#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
PYTHON="/data/home/zhanghx/.conda/envs/PAST/bin/python"

reg_alphas=(
  1.3
  1.4
  1.5
  1.6
)

model="resnet18"
dataset="CIFAR10"
model_root="trained_models/models/${model}/${dataset}"
# No-defense checkpoints are all saved with this legacy filename; the
# architecture is selected by the parent model directory above.
checkpoint_file="resnet18_model.pth"
seed=0
attack_type="white-box"

for reg_alpha in "${reg_alphas[@]}"; do
  training_name="Normal_L1-0.001-10050-10000_${reg_alpha}"
  model_dir="${model_root}/${training_name}"
  target_path="./${model_dir}/target/${checkpoint_file}"
  shadow_path="./${model_dir}/shadow/${checkpoint_file}"
  attack_log_path="./attack_results/models/${model}/${dataset}/${training_name}/Defense/"

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
done
