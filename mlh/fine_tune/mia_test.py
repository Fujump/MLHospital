import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torchvision
from mlh.attacks.membership_inference.attacks import AttackDataset, BlackBoxMIA, MetricBasedMIA, LabelOnlyMIA
from mlh.attacks.membership_inference.data_augmentation_attack import AugemtaionAttackDataset, DataAugmentationMIA
from mlh.models.models_non_image import Purchase,Texas
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
# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_num_threads(1)

# /mnt/sharedata/ssd/users/zhanghx/experiments/mia/ft_defense/exp/CIFAR10/resnet18/target/retrain$

def parse_args():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--training_type', type=str, default="retrain",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE, retrain')
    parser.add_argument('--data_path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--epochs', type=int, default=100)
                        
    parser.add_argument('--epochs_ft', type=int, default=20,
                        help= 'number of training epochs for fine-tuning')
    parser.add_argument('--fine_tune_proportion', type=float, default=0.3,
                        help='proportion of the dataset used for fine-tuning')
    
    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--attack_type', type=str, default='black-box',
                        help='attack type: "black-box", "black-box-sorted", "black-box-top3", "metric-based", and "label-only"')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--log_path', type=str,
                        default='./save', help='')
    
    parser.add_argument('--augment_kwarg_translation', type=float, default=1,
                        help='')
    parser.add_argument('--augment_kwarg_rotation', type=float, default=1,
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--file_path', type=str, default='/mnt/sharedata/ssd/users/zhanghx/experiments/mia/ft_defense/exp',help='model path')
    
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    data_generator = GetDataLoader(args)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = data_generator.get_data_supervised()    
    
    target_path=f'{args.file_path}/{args.dataset}/{args.model}/target/{args.training_type}/{args.fine_tune_proportion}/epoch_{args.epochs}_ft_{args.epochs_ft}.pth'
    shadow_path=f'{args.file_path}/{args.dataset}/{args.model}/shadow/{args.training_type}/{args.fine_tune_proportion}/epoch_{args.epochs}_ft_{args.epochs_ft}.pth'
    target_model=torch.load(target_path)
    shadow_model=torch.load(shadow_path)
    print(f'target_model:{target_path}')
    print(f'shadow_path:{shadow_path}')

    
    target_model = target_model.to(args.device)
    shadow_model = shadow_model.to(args.device)

    # generate attack dataset
    # or "black-box, black-box-sorted", "black-box-top3", "metric-based", and "label-only"
    attack_type = args.attack_type

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
