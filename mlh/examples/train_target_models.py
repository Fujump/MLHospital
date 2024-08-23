import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torchvision
import torch_pruning as tp
from mlh.defenses.membership_inference.AdvReg import TrainTargetAdvReg
from mlh.defenses.membership_inference.DPSGD import TrainTargetDP
from mlh.defenses.membership_inference.LabelSmoothing import TrainTargetLabelSmoothing
from mlh.defenses.membership_inference.MixupMMD import TrainTargetMixupMMD
from mlh.defenses.membership_inference.PATE import TrainTargetPATE
from mlh.defenses.membership_inference.Normal import TrainTargetNormal
from mlh.defenses.membership_inference.pruner import MIAImportance, GradGapPruner
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

    parser.add_argument('--training_type', type=str, default="Normal",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    parser.add_argument('--mode', type=str, default="shadow",
                        help='target, shadow')

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
    parser.add_argument('--task', type=str, default='mia',
                        help='specify the attack task, mia or ol')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--num-class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--inference-dataset', type=str, default='CIFAR10',
                        help='if yes, load pretrained the attack model to inference')
    parser.add_argument('--data-path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--log_path', type=str,
                        default='./save', help='data_path')

    args = parser.parse_args()

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args


def get_target_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(512, 10))
    else:
        raise ValueError("Model not implemented yet :P")
    return model


def evaluate(args, model, dataloader):
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

    opt = parse_args()
    s = GetDataLoader(opt)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised()

    if opt.mode == "target":
        train_loader, inference_loader, test_loader = target_train_loader, target_inference_loader, target_test_loader,
    elif opt.mode == "shadow":
        train_loader, inference_loader, test_loader = shadow_train_loader, shadow_inference_loader, shadow_test_loader
    else:
        raise ValueError("opt.mode should be target or shadow")

    target_model = get_target_model(name="resnet18", num_classes=10)

    if opt.prune=="t":
        save_pth = f'{opt.log_path}/{opt.dataset}/{opt.training_type}_{opt.pruner}_pruned/{opt.mode}' if opt.global_pruning=="f" else f'{opt.log_path}/{opt.dataset}/{opt.training_type}_{opt.pruner}_pruned_global/{opt.mode}'
    else:
        save_pth = f'{opt.log_path}/{opt.dataset}/{opt.training_type}/{opt.mode}'

    if opt.training_type == "Normal":

        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "LabelSmoothing":

        total_evaluator = TrainTargetLabelSmoothing(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "AdvReg":

        total_evaluator = TrainTargetAdvReg(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        model = total_evaluator.model

    elif opt.training_type == "DP":
        total_evaluator = TrainTargetDP(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "MixupMMD":

        target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted = s.get_sorted_data_mixup_mmd()
        if opt.mode == "target":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = target_train_sorted_loader, target_inference_sorted_loader, start_index_target_inference, target_inference_sorted

        elif opt.mode == "shadow":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_shadow_inference, shadow_inference_sorted

        total_evaluator = TrainTargetMixupMMD(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, train_loader_ordered,
                              inference_loader_ordered, test_loader, starting_index, inference_sorted)

    elif opt.training_type == "PATE":

        total_evaluator = TrainTargetPATE(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, inference_loader, test_loader)

    else:
        raise ValueError(
            "opt.training_type should be Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE")
    
    model = target_model
    if opt.prune=="t":
        example_inputs = torch.randn(1, 3, 32, 32).to('cuda')

        # 1. Importance criterion
        if opt.pruner=="norm":
            imp = tp.importance.GroupNormImportance(p=2) # or GroupTaylorImportance(), GroupHessianImportance(), etc.
        elif opt.pruner=="tylor":
            imp = tp.importance.GroupTaylorImportance()
        elif opt.pruner=="hessian":
            imp = tp.importance.GroupHessianImportance()
        elif opt.pruner=="mia":
            # imp = MIAImportance()
            imp = tp.importance.GroupNormImportance(p=1)

        # 2. Initialize a pruner with the model and the importance criterion
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 10:
                ignored_layers.append(m) # DO NOT prune the final classifier!

        pruner = GradGapPruner( # We can always choose MetaPruner if sparse training is not required.
            model,
            example_inputs,
            importance=imp,
            global_pruning=True if opt.global_pruning=="t" else False,
            pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
            ignored_layers=ignored_layers,
        )

        if opt.pruner=="mia":
            total_evaluator = TrainTargetNormal(
                model=target_model, epochs=5, log_path=save_pth)
            total_evaluator.train_sparse(train_loader,inference_loader, test_loader,pruner=pruner)
        
        # 3. Prune & finetune the model
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
        # finetune the pruned model here
        #####################
        from torch.utils.data import DataLoader, random_split
        # 设置随机种子
        torch.manual_seed(42)
        # 获取数据集和数据集的长度
        dataset = train_loader.dataset
        dataset_len = len(dataset)

        # 将数据集平均分成两个
        subset1, subset2 = random_split(dataset, [dataset_len // 2, dataset_len - dataset_len // 2])

        # 为每个子集创建新的 DataLoader
        train_loader_finetune = DataLoader(subset1, batch_size=128, shuffle=True, num_workers=2)
        train_loader_attack = DataLoader(subset2, batch_size=128, shuffle=True, num_workers=2)
        #####################
        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=100, log_path=save_pth)
        total_evaluator.train(train_loader_finetune, test_loader)
        # TODO:如果考虑防御模型，finetune时用对应的防御方法？shadow不用做任何操作？
        # finetune的epoch如何设置，是否要保持总epoch不变？
    
    torch.save(model.state_dict(),
               os.path.join(save_pth, f"{opt.model}.pth"))
    # 4. Save & Load
    model.zero_grad() # clear gradients to avoid a large file size
    torch.save(model,
               os.path.join(save_pth, f"{opt.model}_model.pth")) # !! no .state_dict for saving
    print("Finish Training")
