#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PYTHON="${PYTHON:-/data/home/zhanghx/.conda/envs/PAST/bin/python}"

model="${MODEL:-resnet18}"
dataset="${DATASET:-CIFAR10}"
model_root="${MODEL_ROOT:-/data/home/zhanghx/huq/MLHospital/trained_models/epoch100/${dataset}}"
checkpoint_file="${CHECKPOINT_FILE:-${model}_model.pth}"
seed="${SEED:-0}"
attack_type="${ATTACK_TYPE:-metric-based}"
result_group="${RESULT_GROUP:-epoch100}"
result_root="${RESULT_ROOT:-./attack_results/${result_group}}"

if [[ ! -d "${model_root}" ]]; then
  echo "Model root does not exist: ${model_root}" >&2
  exit 1
fi

mapfile -t model_dirs < <(
  find "${model_root}" -mindepth 1 -maxdepth 1 -type d | sort
)

if [[ "${#model_dirs[@]}" -eq 0 ]]; then
  echo "No model directories found under ${model_root}" >&2
  exit 1
fi

ran_any=0

for model_dir in "${model_dirs[@]}"; do
  training_name="$(basename "${model_dir}")"
  target_path="${model_dir}/target/${checkpoint_file}"
  shadow_path="${model_dir}/shadow/${checkpoint_file}"
  attack_log_path="${result_root}/${model}/${dataset}/${training_name}/Defense/"

  if [[ ! -f "${shadow_path}" ]]; then
    echo "[skip ${training_name}] missing shadow/${checkpoint_file}" >&2
    continue
  fi

  if [[ ! -f "${target_path}" ]]; then
    echo "[skip ${training_name}] missing target/${checkpoint_file}" >&2
    continue
  fi

  echo "[${model} ${training_name}] target=${target_path}"
  echo "[${model} ${training_name}] shadow=${shadow_path}"

  mkdir -p "${attack_log_path}"
  ran_any=1

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

if [[ "${ran_any}" -eq 0 ]]; then
  echo "No complete target/shadow checkpoint pairs found under ${model_root} for ${checkpoint_file}" >&2
  exit 1
fi
