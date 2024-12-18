import torch
import numpy as np
import os
import time
from runx.logx import logx
import torch.nn.functional as F
from mlh.defenses.membership_inference.trainer import Trainer
import torch_pruning as tp
import torch.nn as nn
import argparse

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PruneTuneTargetNormal(Trainer):
    def __init__(
        self,
        model,
        device="cuda",
        num_class=10,
        epochs=100,
        epochs_ft=0,
        learning_rate=0.01,
        learning_rate_ft=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        smooth_eps=0.8,
        log_path="./",
        output_save_path=None,
    ):

        super().__init__()

        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs
        self.epochs_ft = epochs_ft
        self.learning_rate = learning_rate
        self.learning_rate_ft = learning_rate_ft
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.smooth_eps = smooth_eps
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), learning_rate, momentum, weight_decay
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=self.epochs+self.epochs_ft
        # )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        
        self.criterion = nn.CrossEntropyLoss()

        self.log_path = log_path
        logx.initialize(logdir=self.log_path, coolname=False, tensorboard=False)
        # construct the save file path
        self.save_path = output_save_path

    def scheduler_init(self, epochs, learning_rate, momentum, weight_decay):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), learning_rate, momentum, weight_decay)
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        return self.optimizer, self.scheduler
        
    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def eval(self, data_loader):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():

            for img, label in data_loader:
                img, label = img.to(self.device), label.to(self.device)
                logits = self.model.eval().forward(img)

                predicted = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            final_acc = 100 * correct / total

        return final_acc

    def train_one_step(self, data_loader, test_loader, t_start, epoch):
        for img, label in data_loader:
            self.model.zero_grad()
            img, label = img.to(self.device), label.to(self.device)
            logits = self.model(img)
            loss = self.criterion(logits, label)

            loss.backward()
            self.optimizer.step()
        train_acc = self.eval(data_loader)
        test_acc = self.eval(test_loader)
        logx.msg(
                "Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Total Time: %.3fs"
                % (
                    epoch,
                    len(data_loader.dataset),
                    train_acc,
                    test_acc,
                    time.time() - t_start,
                )
            )
        
    def train(self, train_loader, fine_tune_dataset, test_loader, dataset, pruner, global_pruning):
        opt = argparse.Namespace()
        opt.dataset=dataset
        opt.pruner=pruner
        opt.global_pruning=global_pruning
        
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        ### train base model
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_one_step(train_loader, test_loader, t_start, epoch)
            self.scheduler.step()

        self.optimizer, self.scheduler=self.scheduler_init(self.epochs_ft, self.learning_rate_ft, self.momentum, self.weight_decay) # init for fine_tuning
        
        ##########################
        # prune
        if opt.dataset in ["CIFAR10","CIFAR100"]:
            example_inputs = torch.randn(1, 3, 32, 32).to('cuda')
        elif opt.dataset=="texas":
            example_inputs = torch.randn(1, 6169).to('cuda')
        elif opt.dataset=="purchase":
            example_inputs = torch.randn(1, 600).to('cuda')
        elif (opt.dataset=="imagenet") or (opt.dataset=="imagenet_r"):
            example_inputs = torch.randn(1, 3, 224, 224).to('cuda')

        # 1. Importance criterion
        if opt.pruner=="norm":
            imp = tp.importance.GroupNormImportance(p=2) # or GroupTaylorImportance(), GroupHessianImportance(), etc.
        elif opt.pruner=="tylor":
            imp = tp.importance.GroupTaylorImportance()
        elif opt.pruner=="hessian":
            imp = tp.importance.GroupHessianImportance()

        # 2. Initialize a pruner with the model and the importance criterion
        ignored_layers = []
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 10:
                ignored_layers.append(m) # DO NOT prune the final classifier!

        pruner = tp.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
            self.model,
            example_inputs,
            importance=imp,
            global_pruning=True if opt.global_pruning=="t" else False,
            pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
            ignored_layers=ignored_layers,
        )
        
        # 3. Prune & finetune the model
        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
        print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
        ###############################
        
        
        ### fine-tune model
        for epoch in range(1, self.epochs_ft + 1):
            self.model.train()
            self.train_one_step(fine_tune_dataset, test_loader, t_start, epoch)
            self.scheduler.step()

    def check_model_parameters(self, model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN found in {name}")
