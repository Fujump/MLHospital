import sys
import os
import torch
import numpy as np
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))
from mlh.data_preprocessing.data_loader import GetDataLoader
from mlh.attacks.membership_inference.attacks import AttackDataset, BlackBoxMIA, MetricBasedMIA, LabelOnlyMIA
from mlh.attacks.membership_inference.data_augmentation_attack import AugemtaionAttackDataset, DataAugmentationMIA

import argparse
import contextlib
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)

# args=argparse.Namespace(
#     training_type='Reg-0.001-10050-10000_1.5',
#     dataset='CIFAR10',
#     num_class=10,
#     model='resnet18',
#     data_path='../datasets/',
#     log_path='./save',
#     input_shape="32,32,3",
#     device='cuda',
#     attack_type='black-box',
#     batch_size=512,
#     inference_dataset='CIFAR10'
# )
args=argparse.Namespace(
    training_type='Reg-0.001-10050-10000_1.5',attack_type='augmentation', augment_kwarg_rotation=1, augment_kwarg_translation=1, batch_size=512, data_path='../datasets/', dataset='CIFAR10', device='cuda', epochs=100, global_pruning='f', gpu=0, inference_dataset='CIFAR10', input_shape=[32, 32, 3], load_pretrained='no', log_path='./save', model='resnet18', num_class=10, num_workers=10, prune='f', pruner='norm'
)
# args.input_shape = [int(item) for item in args.input_shape.split(',')]

# 初始化模型、损失函数等
s = GetDataLoader(args)
target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised()
member_loader,nonmember_loader=target_train_loader, target_inference_loader

# for inputs, targets in shadow_train_loader:
#             print(f"Inputs shape: {inputs.shape}")
#             print(f"Targets shape: {targets.shape}")
            # break  # 只检查一批数据
# # 打开文件以写入
# with open('metric-based-test.out', 'w') as f:
#     # 重定向标准输出到文件
#     with contextlib.redirect_stdout(f):
#         for i in range(1,101):
#             target_model=torch.load(f'/data/home/huq/MLHospital/log_distribution/loss_gap_mia/target_resnet18_{i}.pth')
#             shadow_model=torch.load(f'/data/home/huq/MLHospital/log_distribution/loss_gap_mia/shadow_resnet18_{i}.pth')

#             attack_type = "metric-based"
#             attack_dataset = AttackDataset(args, attack_type, target_model, shadow_model,
#                                                     target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader)
#             attack_model = MetricBasedMIA(
#                             num_class=args.num_class,
#                             device=args.device,
#                             attack_type=attack_type,
#                             attack_train_dataset=attack_dataset.attack_train_dataset,
#                             attack_test_dataset=attack_dataset.attack_test_dataset,
#                             batch_size=128)
            
#             # 你可以选择在此处添加其他信息输出
#             print(f"Finished iteration {i}\n")  # 例如，这行也会被输出到文件中

# 打开文件以写入
with open('augmentation.out', 'w') as f:
    # 重定向标准输出到文件
    with contextlib.redirect_stdout(f):
        for i in range(1,101):
            target_model=torch.load(f'/data/home/huq/MLHospital/log_distribution/loss_gap_mia/target_resnet18_{i}.pth')
            shadow_model=torch.load(f'/data/home/huq/MLHospital/log_distribution/loss_gap_mia/shadow_resnet18_{i}.pth')

            # target_model=torch.load(f'/data/home/huq/MLHospital/mlh/examples/save/CIFAR10/Reg-0.0001-10050-10000_0.5/target/resnet18_model.pth')
            # shadow_model=torch.load(f'/data/home/huq/MLHospital/mlh/examples/save/CIFAR10/Reg-0.0001-10050-10000_0.5/shadow/resnet18_model.pth')
            
            
            attack_type = "augmentation"
            
            if attack_type == "augmentation":
                attack_dataset_rotation = AugemtaionAttackDataset( args, "rotation" , target_model, shadow_model,
                                                target_train_loader.dataset, target_test_loader.dataset, shadow_train_loader.dataset, shadow_test_loader.dataset,args.device)
                
                attack_dataset_translation =AugemtaionAttackDataset( args, "translation" , target_model, shadow_model,
                                                target_train_loader.dataset, target_test_loader.dataset, shadow_train_loader.dataset, shadow_test_loader.dataset,args.device)
                print(attack_dataset_rotation.attack_train_dataset.data.shape[1])
                print("Attack datasets are ready")
            else:
                attack_dataset = AttackDataset(args, attack_type, target_model, shadow_model,
                                            target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader)
            # print(args)
            # attack_train_loader = torch.utils.data.DataLoader(
            #     attack_dataset.attack_train_dataset, batch_size=128, shuffle=True)
            # for inputs, targets, _ in attack_train_loader:
            #     print(f"Inputs shape: {inputs.shape}")
            #     print(f"Targets shape: {targets.shape}")
            #     break  # 只检查一批数据
            
            # attack_model = BlackBoxMIA(
            #     num_class=args.num_class,
            #     device=args.device,
            #     attack_type=attack_type,
            #     attack_train_dataset=attack_dataset.attack_train_dataset,
            #     attack_test_dataset=attack_dataset.attack_test_dataset,
            #     batch_size=128)
            
            attack_model = DataAugmentationMIA(
                num_class = attack_dataset_rotation.attack_train_dataset.data.shape[1],
                device = args.device, 
                attack_type= "rotation",
                attack_train_dataset=attack_dataset_rotation.attack_train_dataset,  
                attack_test_dataset= attack_dataset_rotation.attack_train_dataset,  
                # save_path= save_path, 
                batch_size= 128)
            attack_model = DataAugmentationMIA(
                num_class = attack_dataset_translation.attack_train_dataset.data.shape[1],
                device = args.device, 
                attack_type= "translation",
                attack_train_dataset=attack_dataset_translation.attack_train_dataset,  
                attack_test_dataset= attack_dataset_translation.attack_test_dataset,
                # save_path= save_path, 
                batch_size= 128)
            
            # 你可以选择在此处添加其他信息输出
            print(f"Finished iteration {i}\n")  # 例如，这行也会被输出到文件中