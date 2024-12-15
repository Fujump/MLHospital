export CUDA_VISIBLE_DEVICES=6

python train_model.py \
    --model resnet18 \
    --dataset CIFAR10 \
    --num_class 10 \
    --training_type retrain \
    --mode shadow \
    --epochs 120 \
    --lr 0.01 \
    --seed 0 \
    --save_path /mnt/sharedata/ssd/users/zhanghx/experiments/mia/ft_defense/exp \
    retrain \
    --epochs_ft 0 \
    --fine_tune_proportion 0.3