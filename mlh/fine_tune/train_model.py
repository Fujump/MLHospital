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
from mlh.defenses.membership_inference.L1 import TrainTargetL1
from mlh.defenses.membership_inference.ConfidencePenalty import TrainTargetConfidencePenalty
from mlh.defenses.membership_inference.Dropout import TrainTargetDropout
from mlh.defenses.membership_inference.PPB import TrainTargetPPB
from mlh.defenses.membership_inference.retrain import RetrainTargetNormal
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
# torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')

    parser.add_argument('--training_type', type=str, default="Normal",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE, retrain')
    
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
    # Parser for Reg
    parser_h = subparsers.add_parser('Reg')
    parser_h.add_argument('--reg_weight', type=float, default=1e-5, help='')
    parser_h.add_argument('--reg_alpha', type=float, default=4, help='')
    parser_h.add_argument('--reg_epoch', type=int, default=50, help='')
    parser_h.add_argument('--reg_clamp', type=int, default=10000, help='')
    parser_h.add_argument('--reg_norm', type=str, default="l1", help='')
    # Parser for L1
    parser_i = subparsers.add_parser('L1')
    parser_i.add_argument('--l1_alpha', type=float, default=0.001, help='')
    # Parser for L2
    parser_j = subparsers.add_parser('L2')
    parser_j.add_argument('--weight_decay', type=float, default=5e-04, help='')
    # Parser for ConfidencePenalty
    parser_k = subparsers.add_parser('ConfidencePenalty')
    parser_k.add_argument('--alpha', type=float, default=0.1, help='')
    # Parser for Dropout
    parser_l = subparsers.add_parser('Dropout')
    parser_l.add_argument('--droprate', type=float, default=0.5, help='')
    # Parser for PPB
    parser_m = subparsers.add_parser('PPB')
    parser_m.add_argument('--ppb_alpha', type=float, default=0.5, help='')
    
    # Parser for retrain
    parser_n = subparsers.add_parser('retrain')
    parser_n.add_argument('--epochs_ft', type=int, default=20,
                        help= 'number of training epochs for fine-tuning')
    parser_n.add_argument('--fine_tune_proportion', type=float, default=0.3,
                        help='proportion of the dataset used for fine-tuning')
    
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
                        default='./save_baseline', help='data_path')
    parser.add_argument('--save_path', type=str, default='./save_baseline',)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument('--model_path', type=str, default= None, help='model path')
    
    args = parser.parse_args()
    
    if args.training_type is None:
        args.training_type = 'Normal'

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
    return model

if __name__ == "__main__":
    opt = parse_args()
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    
    data_generator = GetDataLoader(opt)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = data_generator.get_data_supervised()

    if opt.mode == "target":
        train_loader, inference_loader, test_loader = target_train_loader, target_inference_loader, target_test_loader,
    elif opt.mode == "shadow":
        train_loader, inference_loader, test_loader = shadow_train_loader, shadow_inference_loader, shadow_test_loader
    else:
        raise ValueError("opt.mode should be target or shadow")

    target_model = get_target_model(name=opt.model, num_classes=opt.num_class).cuda()
    save_pth = f'{opt.log_path}/{opt.dataset}/{opt.training_type}/{opt.mode}'
    
    output_save_path = f'{opt.save_path}/{opt.dataset}/{opt.model}/{opt.mode}/{opt.training_type}'

    
    if opt.training_type == "Normal" or opt.training_type == "Reg":
        if opt.training_type == "Reg":
            save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
            if opt.reg_norm=="l1":
                save_pth = f'{save_pth_before_last_slash}-{opt.reg_weight}-{opt.epochs}{opt.reg_epoch}-{opt.reg_clamp}_{opt.reg_alpha}/{save_pth_after_last_slash}'
            else:
                save_pth = f'{save_pth_before_last_slash}-{opt.reg_weight}-{opt.epochs}{opt.reg_epoch}-{opt.reg_clamp}-{opt.reg_norm}_{opt.reg_alpha}/{save_pth_after_last_slash}'
            
        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.epochs, log_path=save_pth, num_class=opt.num_class, weight_decay=opt.weight_l2, output_save_path = output_save_path)
        
        total_evaluator.train(train_loader, inference_loader, test_loader)
      
        
    elif opt.training_type == "PPB":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.ppb_alpha}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetPPB(
            model=target_model, epochs=opt.epochs, log_path=save_pth, num_class=opt.num_class, ppb_alpha=opt.ppb_alpha, weight_decay=opt.weight_l2)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        
    elif opt.training_type == "L2":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.weight_decay}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.epochs, log_path=save_pth, num_class=opt.num_class, weight_decay=opt.weight_decay)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        
    elif opt.training_type == "Dropout":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.droprate}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetDropout(
            model=target_model, epochs=opt.epochs, log_path=save_pth, num_class=opt.num_class, droprate=opt.droprate,model_name=opt.model)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "ConfidencePenalty":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.alpha}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetConfidencePenalty(
            model=target_model, epochs=opt.epochs, log_path=save_pth, num_class=opt.num_class, alpha=opt.alpha)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "CCL":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.ccl_alpha}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetCCL(
            model=target_model, epochs=opt.epochs, log_path=save_pth, alpha=opt.ccl_alpha, num_class=opt.num_class)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "L1":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.l1_alpha}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetL1(
            model=target_model, epochs=opt.epochs, log_path=save_pth, reg_weight=opt.l1_alpha, num_class=opt.num_class, learning_rate=opt.lr, weight_decay=opt.weight_l2)
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
        model = total_evaluator.model

    elif opt.training_type == "DP":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.dp_delta}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetDP(
            model=target_model, epochs=opt.epochs, log_path=save_pth, delta=opt.dp_delta, num_class=opt.num_class)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "MixupMMD":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.mmd_lambda}/{save_pth_after_last_slash}'

        target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted = data_generator.get_sorted_data_mixup_mmd()
        if opt.mode == "target":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = target_train_sorted_loader, target_inference_sorted_loader, start_index_target_inference, target_inference_sorted

        elif opt.mode == "shadow":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_shadow_inference, shadow_inference_sorted

        total_evaluator = TrainTargetMixupMMD(
            model=target_model, epochs=opt.epochs, log_path=save_pth, mixup_alpha=opt.mixup_alpha, mmd_loss_lambda=opt.mmd_lambda, num_class=opt.num_class)
        total_evaluator.train(train_loader, train_loader_ordered,
                              inference_loader_ordered, test_loader, starting_index, inference_sorted)

    elif opt.training_type == "PATE":
        save_pth_before_last_slash, save_pth_after_last_slash = save_pth.rsplit('/', 1)
        save_pth = f'{save_pth_before_last_slash}_{opt.pate_epsilon}/{save_pth_after_last_slash}'

        total_evaluator = TrainTargetPATE(
            model=target_model, epochs=opt.epochs, log_path=save_pth, pate_epsilon=opt.pate_epsilon, num_class=opt.num_class)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        
    elif opt.training_type == "retrain":
        # split dataset for fine-tuning
        fine_tune_dataset = data_generator.get_fine_tune_dataset(train_loader, opt)
        
        total_evaluator = RetrainTargetNormal(
            model=target_model, epochs=opt.epochs, epochs_ft=opt.epochs_ft, log_path=save_pth, output_save_path=output_save_path)
        
        total_evaluator.train(train_loader, fine_tune_dataset, test_loader)
        file_save_path = output_save_path + f"/{opt.fine_tune_proportion}"
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)
        print("Save model to: ", file_save_path)
        
        torch.save(target_model,  file_save_path + f"/epoch_{opt.epochs}_ft_{opt.epochs_ft}.pth")
        
    else:
        raise ValueError(
            "opt.training_type should be Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE, fine_tune")
    
    model = target_model
        
    # if not os.path.exists(output_save_path):
    #     os.makedirs(output_save_path)
    
    # torch.save(model,  output_save_path + f"/epoch_{epochs}.pth")

    
    
    print("Finish Training")
