import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torchvision
from mlh.attacks.membership_inference.attacks import AttackDataset, BlackBoxMIA, MetricBasedMIA, LabelOnlyMIA
from mlh.attacks.membership_inference.data_augmentation_attack import AugemtaionAttackDataset, DataAugmentationMIA
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlh.data_preprocessing.data_loader import GetDataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.optim as optim

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def select_loader_samples(loader, sample_size=5000, seed=0):
    dataset = loader.dataset
    if len(dataset) <= sample_size:
        return loader

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_size].tolist()
    subset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )
    
    
def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index used for training')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--num-class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--training_type', type=str, default="Normal",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    parser.add_argument('--attack_type', type=str, default='black-box',
                        help='attack type: "black-box", "black-box-sorted", "black-box-top3", "metric-based", and "label-only"')
    parser.add_argument('--data-path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--attack_log_path', type=str,
                        default='./save', help='')
    parser.add_argument('--target_path', type=str, default=None,
                        help='path to target model checkpoint')
    parser.add_argument('--shadow_path', type=str, default=None,
                        help='path to shadow model checkpoint')
    parser.add_argument('--sample_size', type=int, default=5000,
                        help='number of samples selected from target/shadow train and test loaders')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    
    parser.add_argument('--augment_kwarg_translation', type=float, default=1,
                        help='')
    parser.add_argument('--augment_kwarg_rotation', type=float, default=1,
                        help='')

    args = parser.parse_args()

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    # args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args


if __name__ == "__main__":

    args = parse_args()
    set_seed(args.seed)
    s = GetDataLoader(args)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised()
  
    # if args.prune=="t":
    #     t_path=f'{args.log_path}/{args.dataset}/{args.training_type}_{args.pruner}_pruned/target/{args.model}_model.pth' if args.global_pruning=="f" else f'{args.log_path}/{args.dataset}/{args.training_type}_{args.pruner}_pruned_global/target/{args.model}_model.pth'
    #     s_path=f'{args.log_path}/{args.dataset}/{args.training_type}_{args.pruner}_pruned/shadow/{args.model}_model.pth' if args.global_pruning=="f" else f'{args.log_path}/{args.dataset}/{args.training_type}_{args.pruner}_pruned_global/shadow/{args.model}_model.pth'
    #     target_model=torch.load(t_path)
    #     shadow_model=torch.load(s_path)
    # else:
    
    # target_model=torch.load(f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.model}_model.pth')
    # shadow_model=torch.load(f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.model}_model.pth')
    # target_path=f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.model}_model.pth'
    # shadow_path=f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.model}_model.pth'
    # print(f'target_model:{target_path}')
    # print(f'shadow_path:{shadow_path}')
    
    if args.target_path is None or args.shadow_path is None:
        raise ValueError("--target_path and --shadow_path are required")

    target_path = args.target_path
    shadow_path = args.shadow_path
    # target_path = "/data/home/zhanghx/huq/MLHospital/trained_models/epoch100/CIFAR10/Normal_L1-0.001-10050-10000_1.1/target/resnet18_model.pth"
    # shadow_path = "/data/home/zhanghx/huq/MLHospital/trained_models/epoch100/CIFAR10/Normal_L1-0.001-10050-10000_1.1/shadow/resnet18_model.pth"
    
    print(f'target_model:{target_path}')
    print(f'shadow_path:{shadow_path}')
    target_model = torch.load(target_path)
    shadow_model = torch.load(shadow_path)
    target_model = target_model.to(args.device)
    shadow_model = shadow_model.to(args.device)
    target_model.eval()
    shadow_model.eval()

    # generate attack dataset
    # or "black-box, black-box-sorted", "black-box-top3", "metric-based", and "label-only"
    
    attack_type = args.attack_type

    # target_train_loader = select_loader_samples(target_train_loader, sample_size=args.sample_size, seed=args.seed)
    # shadow_train_loader = select_loader_samples(shadow_train_loader, sample_size=args.sample_size, seed=args.seed)
    # target_test_loader = select_loader_samples(target_test_loader, sample_size=args.sample_size, seed=args.seed)
    # shadow_test_loader = select_loader_samples(shadow_test_loader, sample_size=args.sample_size, seed=args.seed)
    
    target_train_loader = select_loader_samples(target_inference_loader, sample_size=args.sample_size, seed=args.seed)
    shadow_train_loader = select_loader_samples(shadow_inference_loader, sample_size=args.sample_size, seed=args.seed)
    
    target_test_loader = select_loader_samples(target_test_loader, sample_size=args.sample_size, seed=args.seed)
    shadow_test_loader = select_loader_samples(shadow_test_loader, sample_size=args.sample_size, seed=args.seed)
    
    
    # target_test_loader = select_loader_samples(target_inference_loader, sample_size=args.sample_size, seed=args.seed)
    # shadow_test_loader = select_loader_samples(shadow_inference_loader, sample_size=args.sample_size, seed=args.seed)
    

    
    print(f"train_loader: {len(target_train_loader.dataset)} samples")
    print(f"inference_loader: {len(shadow_train_loader.dataset)} samples")
        
    if attack_type == "label-only":
        attack_model = LabelOnlyMIA(
            device=args.device,
            target_model=target_model.eval(),
            shadow_model=shadow_model.eval(),
            target_loader=(target_train_loader, target_test_loader),
            shadow_loader=(shadow_train_loader, shadow_test_loader),
            input_shape=(3, 32, 32),
            nb_classes=10)
        auc = attack_model.Infer()
        print(auc)

    else:
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

        # train attack model

        if "black-box" in attack_type:
            attack_model = BlackBoxMIA(
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                batch_size=128)
        elif ("metric-based" in attack_type) or ("white-box" in attack_type):
            attack_model = MetricBasedMIA(
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                batch_size=128)
        elif "augmentation" in attack_type:
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
