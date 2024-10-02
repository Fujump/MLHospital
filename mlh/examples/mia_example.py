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
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)


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

    # pruning
    parser.add_argument('--prune', type=str, default="f",
                        help='t(true), f(false)')
    parser.add_argument('--pruner', type=str, default="norm",
                        help='norm, tylor, hessian, mia')
    parser.add_argument('--global_pruning', type=str, default="f",
                        help='t(true), f(false)')
    
    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--num-class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--training_type', type=str, default="Normal",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    parser.add_argument('--inference-dataset', type=str, default='CIFAR10',
                        help='if yes, load pretrained attack model to inference')
    parser.add_argument('--attack_type', type=str, default='black-box',
                        help='attack type: "black-box", "black-box-sorted", "black-box-top3", "metric-based", and "label-only"')
    parser.add_argument('--data-path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--log_path', type=str,
                        default='./save', help='')
    
    parser.add_argument('--augment_kwarg_translation', type=float, default=1,
                        help='')
    parser.add_argument('--augment_kwarg_rotation', type=float, default=1,
                        help='')

    args = parser.parse_args()

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    # args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args


def get_target_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(512, 10))
    elif name == "dense121":
        model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        # model = torchvision.models.densenet121()
        model.classifier = nn.Sequential(nn.Linear(1024, num_classes))
    elif name == "TexasClassifier":
        model= Texas(num_classes = num_classes)
    elif name == "PurchaseClassifier":
        model= Purchase(num_classes = num_classes)

    else:
        raise ValueError("Model not implemented yet :P")
    return model

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    return correct / total


if __name__ == "__main__":

    args = parse_args()
    s = GetDataLoader(args)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised()
    # #####################
    # from torch.utils.data import DataLoader, random_split
    # # 设置随机种子
    # torch.manual_seed(42)
    # # 获取数据集和数据集的长度
    # dataset = target_train_loader.dataset
    # dataset_len = len(dataset)

    # # 将数据集平均分成两个
    # subset1, subset2 = random_split(dataset, [dataset_len // 2, dataset_len - dataset_len // 2])

    # # 为每个子集创建新的 DataLoader
    # train_loader_finetune = DataLoader(subset1, batch_size=128, shuffle=True, num_workers=2)
    # train_loader_attack = DataLoader(subset2, batch_size=128, shuffle=True, num_workers=2)
    
    # target_train_loader=train_loader_attack
    # #########################
    # # 获取数据集和数据集的长度
    # dataset = target_test_loader.dataset
    # dataset_len = len(dataset)

    # # 将数据集平均分成两个
    # subset1, subset2 = random_split(dataset, [dataset_len // 2, dataset_len - dataset_len // 2])

    # # 为每个子集创建新的 DataLoader
    # test_loader_1 = DataLoader(subset1, batch_size=128, shuffle=True, num_workers=2)
    # test_loader_2 = DataLoader(subset2, batch_size=128, shuffle=True, num_workers=2)
    
    # target_test_loader=test_loader_1
    # #####################
    # #####################
    # from torch.utils.data import DataLoader, random_split
    # # 设置随机种子
    # torch.manual_seed(42)
    # # 获取数据集和数据集的长度
    # dataset = shadow_train_loader.dataset
    # dataset_len = len(dataset)

    # # 将数据集平均分成两个
    # subset1, subset2 = random_split(dataset, [dataset_len // 2, dataset_len - dataset_len // 2])

    # # 为每个子集创建新的 DataLoader
    # train_loader_finetune = DataLoader(subset1, batch_size=128, shuffle=True, num_workers=2)
    # train_loader_attack = DataLoader(subset2, batch_size=128, shuffle=True, num_workers=2)
    
    # shadow_train_loader=train_loader_attack
    # #########################
    # # 获取数据集和数据集的长度
    # dataset = shadow_test_loader.dataset
    # dataset_len = len(dataset)

    # # 将数据集平均分成两个
    # subset1, subset2 = random_split(dataset, [dataset_len // 2, dataset_len - dataset_len // 2])

    # # 为每个子集创建新的 DataLoader
    # test_loader_1 = DataLoader(subset1, batch_size=128, shuffle=True, num_workers=2)
    # test_loader_2 = DataLoader(subset2, batch_size=128, shuffle=True, num_workers=2)
    
    # shadow_test_loader=test_loader_1
    # #####################
    # target = get_target_model(name=args.model, num_classes=args.num_class)
    # shadow = get_target_model(name=args.model, num_classes=args.num_class)

    # # load target/shadow model to conduct the attacks
    # target_model.load_state_dict(torch.load(
    #     f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.model}.pth'))
    # target_model = target_model.to(args.device)

    # shadow_model.load_state_dict(torch.load(
    #     f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.model}.pth'))
    # shadow_model = shadow_model.to(args.device)
    if args.prune=="t":
        t_path=f'{args.log_path}/{args.dataset}/{args.training_type}_{args.pruner}_pruned/target/{args.model}_model.pth' if args.global_pruning=="f" else f'{args.log_path}/{args.dataset}/{args.training_type}_{args.pruner}_pruned_global/target/{args.model}_model.pth'
        s_path=f'{args.log_path}/{args.dataset}/{args.training_type}_{args.pruner}_pruned/shadow/{args.model}_model.pth' if args.global_pruning=="f" else f'{args.log_path}/{args.dataset}/{args.training_type}_{args.pruner}_pruned_global/shadow/{args.model}_model.pth'
        target_model=torch.load(t_path)
        shadow_model=torch.load(s_path)
    else:
        target_model=torch.load(f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.model}_model.pth')
        shadow_model=torch.load(f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.model}_model.pth')
        target_path=f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.model}_model.pth'
        shadow_path=f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.model}_model.pth'
        print(f'target_model:{target_path}')
        print(f'shadow_path:{shadow_path}')
    
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in target_model['state_dict'].items():
    #     name = k.replace('module.', '')  # 去掉 `module.` 前缀
    #     new_state_dict[name] = v
    # # 加载模型参数
    # target.load_state_dict(new_state_dict)
    # new_state_dict = OrderedDict()
    # for k, v in shadow_model['state_dict'].items():
    #     name = k.replace('module.', '')  # 去掉 `module.` 前缀
    #     new_state_dict[name] = v
    # # 加载模型参数
    # shadow.load_state_dict(new_state_dict)
    # target_model,shadow_model=target,shadow
    
    
    target_model = target_model.to(args.device)
    shadow_model = shadow_model.to(args.device)

    # generate attack dataset
    # or "black-box, black-box-sorted", "black-box-top3", "metric-based", and "label-only"
    attack_type = args.attack_type

    # attack_type = "metric-based"

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
