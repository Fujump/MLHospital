import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from mlh.attacks.membership_inference.attacks import AttackDataset, MetricBasedMIA
from mlh.data_preprocessing.data_loader import GetDataLoader


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser('argument for top loss leakage attack')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--num-class', type=int, default=10)
    parser.add_argument('--attack_type', type=str, default='metric-based')
    parser.add_argument('--data-path', type=str, default='../datasets/')
    parser.add_argument('--input-shape', type=str, default="32,32,3")
    parser.add_argument('--attack_log_path', type=str, default='./save')
    parser.add_argument('--target_path', type=str, default=None)
    parser.add_argument('--shadow_path', type=str, default=None)
    parser.add_argument('--member-samples-path', type=str,
                        default='./attack_results/test/target_loss_leakage_samples.pt')
    parser.add_argument('--member-count', type=int, default=500)
    parser.add_argument('--nonmember-count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def build_member_loader(args):
    leakage_data = torch.load(args.member_samples_path, map_location='cpu')
    samples = leakage_data["samples"][:args.member_count].float()
    labels = leakage_data["target_labels"][:args.member_count].long()
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    return loader


def build_nonmember_loader(dataset, args):
    if args.nonmember_count > len(dataset):
        raise ValueError(f"--nonmember-count {args.nonmember_count} exceeds target test size {len(dataset)}")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:args.nonmember_count].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    return loader, indices


if __name__ == "__main__":
    args = parse_args()
    if args.attack_type != "metric-based":
        raise ValueError("mia_loss_top500_example.py only supports --attack_type metric-based")
    if args.target_path is None or args.shadow_path is None:
        raise ValueError("--target_path and --shadow_path are required")

    set_seed(args.seed)
    os.makedirs(args.attack_log_path, exist_ok=True)

    data_loader = GetDataLoader(args)
    _, _, target_test_loader, shadow_train_loader, _, shadow_test_loader = data_loader.get_data_supervised(
        batch_size=args.batch_size, num_workers=args.num_workers)

    member_loader = build_member_loader(args)
    nonmember_loader, nonmember_indices = build_nonmember_loader(target_test_loader.dataset, args)
    torch.save({"target_test_indices": nonmember_indices},
               os.path.join(args.attack_log_path, "sampled_nonmember_indices.pt"))

    print(f"member samples: {len(member_loader.dataset)}")
    print(f"nonmember samples: {len(nonmember_loader.dataset)}")
    print(f'target_model:{args.target_path}')
    print(f'shadow_path:{args.shadow_path}')

    target_model = torch.load(args.target_path, map_location=args.device)
    shadow_model = torch.load(args.shadow_path, map_location=args.device)
    target_model = target_model.to(args.device).eval()
    shadow_model = shadow_model.to(args.device).eval()

    attack_dataset = AttackDataset(args, args.attack_type, target_model, shadow_model,
                                   member_loader, nonmember_loader, shadow_train_loader, shadow_test_loader)
    MetricBasedMIA(
        num_class=args.num_class,
        device=args.device,
        attack_type=args.attack_type,
        attack_train_dataset=attack_dataset.attack_train_dataset,
        attack_test_dataset=attack_dataset.attack_test_dataset,
        batch_size=args.batch_size,
        save_path=args.attack_log_path,
        target_train_samples=attack_dataset.target_train_info.get("samples"))
