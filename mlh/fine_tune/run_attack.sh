export CUDA_VISIBLE_DEVICES=7

python mia_test.py \
    --training_type retrain \
    --model resnet18 \
    --dataset CIFAR10 \
    --num_class 10 \
    --epochs 120 \
    --epochs_ft 0 \
    --fine_tune_proportion 0.3 \
    --seed 0 \
    --attack_type metric-based \
    --file_path /mnt/sharedata/ssd/users/zhanghx/experiments/mia/ft_defense/exp 