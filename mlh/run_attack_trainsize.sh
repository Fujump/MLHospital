#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
PYTHON="/data/home/zhanghx/.conda/envs/PAST/bin/python"

sample_sizes=(
  10000
)
seed=0
attack_type="metric-based"
# trained_models/epoch100/CIFAR10/Normal_L1-0.001-10050-10000_1.1/shadow/model_epochs_0.0/resnet18_100.pth
for sample_size in "${sample_sizes[@]}"; do
  # model_dir="/data/home/zhanghx/huq/MLHospital/trained_models/trainsize/${sample_size}/CIFAR10/Normal_L1-0.001-10050-10000_1.3"
  # model_dir="/data/home/zhanghx/huq/MLHospital/trained_models/epoch100/CIFAR10/Normal_L1-0.001-10050-10000_1.1"
  model_dir="/data/home/zhanghx/huq/MLHospital/trained_models/epoch100/seed_0/CIFAR10/Normal_L1-0.001-10050-10000_1.4"
  target_path="${model_dir}/target/resnet18_model.pth"
  shadow_path="${model_dir}/shadow/resnet18_model.pth"
  attack_log_path="./attack_results/trainsize/${sample_size}/past_inference/"

  mkdir -p "${attack_log_path}"

  "${PYTHON}" mlh/examples/mia_trainsize_example.py \
    --gpu 0 \
    --model resnet18 \
    --dataset CIFAR10 \
    --num-class 10 \
    --attack_log_path "${attack_log_path}" \
    --attack_type "${attack_type}" \
    --batch-size 128 \
    --target_path "${target_path}" \
    --shadow_path "${shadow_path}" \
    --sample_size "${sample_size}" \
    --seed "${seed}" \
    2>&1 | tee "${attack_log_path}/${attack_type}.log"
done
