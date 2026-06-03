import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torchvision
from mlh.defenses.membership_inference.AdvReg import TrainTargetAdvReg
from mlh.defenses.membership_inference.DPSGD import TrainTargetDP
from mlh.defenses.membership_inference.LabelSmoothing import TrainTargetLabelSmoothing
from mlh.defenses.membership_inference.MixupMMD import TrainTargetMixupMMD
from mlh.defenses.membership_inference.PATE import TrainTargetPATE
from mlh.defenses.membership_inference.Normal import TrainTargetNormal
from mlh.defenses.membership_inference.RelaxLoss import TrainTargetRelaxLoss
from mlh.defenses.membership_inference.CCL import TrainTargetCCL
from mlh.defenses.membership_inference.pruner import PAST
from mlh.models.models_non_image import Purchase,Texas
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlh.data_preprocessing.data_loader import GetDataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.optim as optim
import gc

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')
    parser.add_argument('--training_type', dest='training_type_arg', default=None,
                        help='legacy training type argument; prefer the subcommand form')
    parser.add_argument('--reg_weight', type=float, default=1e-5, help='')
    parser.add_argument('--reg_alpha', type=float, default=4, help='')
    parser.add_argument('--reg_epoch', type=int, default=50, help='')
    parser.add_argument('--reg_clamp', type=int, default=10000, help='')
    parser.add_argument('--reg_norm', type=str, default="l1", help='')

    # parser.add_argument('--training_type', type=str, default="Normal",
    #                     help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    subparsers = parser.add_subparsers(dest='training_type', required=False)
    # Parser for LabelSmoothing
    parser_a = subparsers.add_parser('LabelSmoothing')
    parser_a.add_argument('--smooth_eps', type=float, default=0.8, help='')
    # Parser for AdvReg
    parser_b = subparsers.add_parser('AdvReg')
    parser_b.add_argument('--adv_alpha', type=float, default=1, help='')
    # Parser for DP
    parser_c = subparsers.add_parser('DP')
    parser_c.add_argument('--dp_delta', type=float, default=1e-5, help='')
    # Parser for MixupMMD
    parser_d = subparsers.add_parser('MixupMMD')
    parser_d.add_argument('--mixup_alpha', type=float, default=1.0, help='')
    parser_d.add_argument('--mmd_lambda', type=float, default=3, help='')
    # Parser for PATE
    parser_e = subparsers.add_parser('PATE')
    parser_e.add_argument('--pate_epsilon', type=float, default=0.2, help='')
    # Parser for RelaxLoss
    parser_f = subparsers.add_parser('RelaxLoss')
    parser_f.add_argument('--relax_alpha', type=float, default=1, help='')
    # Parser for CCL
    parser_g = subparsers.add_parser('CCL')
    parser_g.add_argument('--ccl_alpha', type=float, default=0.5, help='')
    # Parser for PAST
    parser_h = subparsers.add_parser('PAST', aliases=['past', 'Reg'])
    parser_h.add_argument('--reg_weight', type=float, default=1e-5, help='')
    parser_h.add_argument('--reg_alpha', type=float, default=4, help='')
    parser_h.add_argument('--reg_epoch', type=int, default=50, help='')
    parser_h.add_argument('--reg_clamp', type=int, default=10000, help='')
    parser_h.add_argument('--reg_norm', type=str, default="l1", help='')
    parser_h.add_argument('--reg_lr', type=float, default=0.01, help='')
    
    # pre-train
    parser.add_argument('--pre_train', type=str, default="Normal",
                        help='')
    
    parser.add_argument('--mode', type=str, default="shadow",
                        help='target, shadow')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--weight_l2', type=float, default=5e-04, help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
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
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--inference-dataset', type=str, default='CIFAR10',
                        help='if yes, load pretrained the attack model to inference')
    parser.add_argument('--data-path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--log_path', type=str,
                        default='./save', help='data_path')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()
    
    args.training_type = args.training_type or args.training_type_arg
    if args.training_type is None:
        args.training_type = 'Normal'
    elif args.training_type in ["past", "Reg"]:
        args.training_type = "PAST"
    del args.training_type_arg

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    # args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def get_target_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        if num_classes==100:
            model = torchvision.models.resnet18(pretrained=True)
        else:
            model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(512, num_classes))
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
    print("model name:", name)
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
        if np.isnan(np.sum(predicted)) or np.isnan(np.sum(outputs)):
            raise ValueError("Input contains NaN values.")
        correct += predicted.eq(labels).sum().item()
    model.train()
    return correct / total


if __name__ == "__main__":

    opt = parse_args()
    set_seed(opt.seed)
    s = GetDataLoader(opt)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised(batch_size=128, num_workers=0)

    if opt.mode == "target":
        train_loader, inference_loader, test_loader = target_train_loader, target_inference_loader, target_test_loader,
    elif opt.mode == "shadow":
        train_loader, inference_loader, test_loader = shadow_train_loader, shadow_inference_loader, shadow_test_loader
    else:
        raise ValueError("opt.mode should be target or shadow")

    if opt.mode=='target':
        target_path = "trained_models/CIFAR10/Normal_L1-0.001-10050-10000_1.4/target/model_epochs_0.0/resnet18_100.pth"
        target_model=torch.load(target_path)
    elif opt.mode=='shadow':
        shadow_path = "trained_models/CIFAR10/Normal_L1-0.001-10050-10000_1.4/shadow/model_epochs_0.0/resnet18_100.pth"
        target_model=torch.load(shadow_path)
        # target_model=torch.load(f'{opt.log_path}/{opt.dataset}/{opt.pre_train}/shadow/{opt.model}_model.pth')
    
    # target_model = get_target_model(name=opt.model, num_classes=opt.num_class).cuda()
     
    save_pth = f'{opt.log_path}/{opt.dataset}/{opt.pre_train}_L1/{opt.mode}'

    if opt.training_type == "PAST":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        if opt.reg_norm=="l1":
            save_pth = f'{save_pth_before_last_slash}-{opt.reg_weight}-{opt.epochs}{opt.reg_epoch}-{opt.reg_clamp}_{opt.reg_alpha}_{opt.reg_lr}/{save_pth_after_last_slash}'
        else:
            save_pth = f'{save_pth_before_last_slash}-{opt.reg_weight}-{opt.epochs}{opt.reg_epoch}-{opt.reg_clamp}-{opt.reg_norm}_{opt.reg_alpha}_{opt.reg_lr}/{save_pth_after_last_slash}'

        # total_evaluator = TrainTargetNormal(
        # model=target_model, epochs=opt.epochs, learning_rate=opt.lr, log_path=save_pth, num_class=opt.num_class, weight_decay=opt.weight_l2)
        # total_evaluator.train(train_loader, inference_loader, test_loader)
        
    elif opt.training_type == "Normal":
        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.epochs, learning_rate=opt.lr, log_path=save_pth, num_class=opt.num_class, weight_decay=opt.weight_l2)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        
    elif opt.training_type == "CCL":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.ccl_alpha}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetCCL(
            model=target_model, epochs=opt.epochs, log_path=save_pth, alpha=opt.ccl_alpha, num_class=opt.num_class)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "RelaxLoss":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.relax_alpha}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetRelaxLoss(
            model=target_model, epochs=opt.epochs, log_path=save_pth, alpha=opt.relax_alpha, num_class=opt.num_class)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "LabelSmoothing":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.smooth_eps}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetLabelSmoothing(
            model=target_model, epochs=opt.epochs, log_path=save_pth, smooth_eps=opt.smooth_eps, num_class=opt.num_class)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "AdvReg":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.adv_alpha}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetAdvReg(
            model=target_model, epochs=opt.epochs, log_path=save_pth, alpha=opt.adv_alpha, num_class=opt.num_class)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        # model = total_evaluator.model

    elif opt.training_type == "DP":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.dp_delta}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetDP(
            model=target_model, epochs=opt.epochs, log_path=save_pth, delta=opt.dp_delta, num_class=opt.num_class)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "MixupMMD":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.mmd_lambda}/{save_pth_after_last_slash}'

    else:
        raise ValueError(
            "opt.training_type should be Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE")
    
    model = target_model

    if opt.training_type == "PAST":
        pruner = PAST()
        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.reg_epoch, learning_rate=opt.reg_lr, weight_decay=0, log_path=save_pth)
        
        total_evaluator.train_sparse(train_loader,inference_loader, test_loader,pruner=pruner,args=opt)
    
    del target_train_loader, target_inference_loader, target_test_loader
    del shadow_train_loader, shadow_inference_loader, shadow_test_loader
    del train_loader, inference_loader, test_loader
    gc.collect()

    model = model.to("cpu")
    torch.cuda.empty_cache()

    torch.save(model.state_dict(),
               os.path.join(save_pth, f"{opt.model}.pth"))
    # 4. Save & Load
    model.zero_grad() # clear gradients to avoid a large file size
    torch.save(model,
               os.path.join(save_pth, f"{opt.model}_model.pth")) # !! no .state_dict for saving
    print("Finish Training", flush=True)
