# MIT License

# Copyright (c) 2022 The Machine Learning Hospital (MLH) Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
import os
import time
from runx.logx import logx
import torch.nn.functional as F
from mlh.defenses.membership_inference.trainer import Trainer
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# class LabelSmoothingLoss(torch.nn.Module):
#     """
#     copy from:
#     https://github.com/pytorch/pytorch/issues/7455
#     """

#     def __init__(self, classes, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TrainTargetNormal(Trainer):
    def __init__(self, model, device="cuda", num_class=10, epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, smooth_eps=0.8, log_path="./"):

        super().__init__()

        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs
        self.smooth_eps = smooth_eps

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), learning_rate, momentum, weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

        self.criterion = nn.CrossEntropyLoss()

        # self.log_path = "%smodel_%s_bs_%s_dataset_%s/%s/label_smoothing_%.1f" % (self.opt.model_save_path, self.opt.model,
        # #                                                                           self.opt.batch_size, self.opt.dataset, self.opt.mode, self.opt.smooth_eps)
        # self.model_save_name = 'model_%s_label_smoothing_%.1f' % (
        #     self.opt.mode, self.opt.smooth_eps)

        # logx.initialize(logdir=self.log_path,
        #                 coolname=False, tensorboard=False)

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

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

    def train(self, train_loader, nonmember_loader, test_loader):

        best_accuracy = 0
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, '%s_0.pth' % (self.model_save_name)))

        for e in range(1, self.epochs+1):
            batch_n = 0
            self.model.train()
            #############
            member_iter = iter(train_loader)
            nonmember_iter = iter(nonmember_loader)
            
            for img, label in train_loader:
                self.model.zero_grad()
                batch_n += 1

                img, label = img.to(self.device), label.to(self.device)

                # 获取对应的 nonmember 数据
                try:
                    nonmember_img, nonmember_label = next(nonmember_iter)
                except StopIteration:
                    nonmember_iter = iter(nonmember_loader)
                    nonmember_img, nonmember_label = next(nonmember_iter)

                nonmember_img = nonmember_img.to(self.device)
                nonmember_label = nonmember_label.to(self.device)
                
                # 计算 nonmember 数据的梯度（首先计算 nonmember 的梯度）
                logits_nonmember = self.model(nonmember_img)
                loss_nonmember = self.criterion(logits_nonmember, nonmember_label) # 计算非成员数据的损失
                loss_nonmember.backward(retain_graph=True)
                # nonmember_grads = {m.weight: m.weight.grad.clone() for m in self.model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine}
                nonmember_grads = {name: m.weight.grad.clone() for name, m in self.model.named_modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear))}
                # nonmember_grads = {
                #     name: param.grad.clone() 
                #     for name, param in self.model.named_parameters()
                #     if isinstance(self.model._modules[name], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear))
                # }

                # 计算 member 数据的梯度（然后计算 member 的梯度，保留在模型中的梯度将基于 member 数据）
                self.model.zero_grad()  # 重置梯度
                logits_member = self.model(img)
                loss_member = self.criterion(logits_member, label)
                loss_member.backward(retain_graph=True)
                
                if (batch_n%100==1):
                    loss_gap = loss_member-loss_nonmember
                    # torch.save(loss_gap, f"/data/home/huq/MLHospital/log_distribution/loss_gap_mia/loss_gap_{e}_{batch_n}.pth")
                
                self.optimizer.step()
            #############
            
            # for img, label in train_loader:
            #     self.model.zero_grad()
            #     batch_n += 1

            #     img, label = img.to(self.device), label.to(self.device)
            #     # print("img", img.shape)
            #     logits = self.model(img)
            #     loss = self.criterion(logits, label)

            #     loss.backward()
            #     self.optimizer.step()

            train_acc = self.eval(train_loader)
            test_acc = self.eval(test_loader)
            logx.msg('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Total Time: %.3fs' % (
                e, len(train_loader.dataset), train_acc, test_acc, time.time() - t_start))
            self.scheduler.step()
            # posteriors = F.softmax(logits, dim=1)
            # print("prediction posteriors", posteriors[0])
            # if e % 10 == 0:
            #     torch.save(self.model.state_dict(), os.path.join(
            #         self.log_path, '%s_%d.pth' % (self.model_save_name, e)))
            torch.save(self.model,f"/data/home/huq/MLHospital/log_distribution/loss_gap_mia/shadow_resnet18_{e}.pth")
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, "%s.pth" % self.model_save_name))

    def check_model_parameters(self, model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN found in {name}")

    def train_sparse(self, member_loader, nonmember_loader, test_loader, pruner,args=None):
        # self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        best_accuracy = 0
        train_acc=0
        test_acc=0
        t_start = time.time()

        # check whether path exists
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        for e in range(1, self.epochs + 1):
            batch_n = 0
            self.model.train()
            member_iter = iter(member_loader)
            nonmember_iter = iter(nonmember_loader)
            
            for img, label in member_loader:
                self.model.zero_grad()
                batch_n += 1

                img, label = img.to(self.device), label.to(self.device)

                # 获取对应的 nonmember 数据
                try:
                    nonmember_img, nonmember_label = next(nonmember_iter)
                except StopIteration:
                    nonmember_iter = iter(nonmember_loader)
                    nonmember_img, nonmember_label = next(nonmember_iter)

                nonmember_img = nonmember_img.to(self.device)
                nonmember_label = nonmember_label.to(self.device)
                
                # 计算 nonmember 数据的梯度（首先计算 nonmember 的梯度）
                logits_nonmember = self.model(nonmember_img)
                loss_nonmember = self.criterion(logits_nonmember, nonmember_label) # 计算非成员数据的损失
                # loss_nonmember.backward(retain_graph=True)
                # # nonmember_grads = {m.weight: m.weight.grad.clone() for m in self.model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine}
                # nonmember_grads = {name: m.weight.grad.clone() for name, m in self.model.named_modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear))}
                # nonmember_grads = {
                #     name: param.grad.clone() 
                #     for name, param in self.model.named_parameters()
                #     if isinstance(self.model._modules[name], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear))
                # }

                # 计算 member 数据的梯度（然后计算 member 的梯度，保留在模型中的梯度将基于 member 数据）
                self.model.zero_grad()  # 重置梯度
                logits_member = self.model(img)
                loss_member = self.criterion(logits_member, label)
                loss_gap = torch.abs(loss_member-loss_nonmember)
                loss_gap.backward(retain_graph=True)
                loss_gap_grads = {name: m.weight.grad.clone() for name, m in self.model.named_modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear))}
                
                self.model.zero_grad()
                loss_member.backward(retain_graph=True)
                # member_grads = {m.weight: m.weight.grad.clone() for m in self.model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine}
                # member_grads = {name: m.weight.grad.clone() for name, m in self.model.named_modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear))}
                # member_grads = {
                #     name: param.grad.clone() 
                #     for name, param in self.model.named_parameters()
                #     if isinstance(self.model._modules[name], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear))
                # }

                # 计算梯度差异（grad gap）
                # grad_gaps = pruner.compute_grad_gap(member_grads, nonmember_grads)
                # # # if (e%5==1) and (batch_n%100==1):
                # # if (batch_n%100==1):
                # # # Compute loss gap
                # loss_gap = torch.abs(loss_member-loss_nonmember)

                # # Compute loss gap derivative for each parameter
                # loss_gap_grads = {
                #     name: torch.autograd.grad(loss_gap, m.weight, retain_graph=True)[0]
                #     for name, m in self.model.named_modules()
                #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear))
                # }
                    
                    # # torch.save(loss_gap, f"/data/home/huq/MLHospital/log_distribution/loss_gap_grad_noabs/loss_gap_{e}_{batch_n}.pth")
                    
                    # grad_gaps_noabs = {name: (member_grads[name] - nonmember_grads[name]) for name in member_grads.keys()}
                    # # torch.save(grad_gaps, f'/data/home/huq/MLHospital/log_distribution/grad_gaps_normal_noabs/grad_gaps_normal_noabs_{e}_{batch_n}.pth')

                # if e>1 and test_acc<best_accuracy/2:
                #     if (batch_n%100==1):
                #         print("improve acc, no regularization")
                # else:
                # 执行自适应正则化
                args.optimizer=self.optimizer
                pruner.regularize(self.model, loss_gap_grads, reg_weight=args.reg_weight, adaptive_strength=args.reg_alpha, args=args,e=e,batch_n=batch_n)
                # print("sparse training")
                # 更新模型参数
                self.optimizer.step()
                # self.check_model_parameters(self.model)

            # torch.save(self.model,f"/data/home/huq/MLHospital/log_distribution/model_epochs_0.0/resnet18_{e}.pth")
            train_acc = self.eval(member_loader)
            test_acc = self.eval(test_loader)
            if test_acc>best_accuracy:
                best_accuracy=test_acc
            logx.msg('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Total Time: %.3fs' % (
                e, len(member_loader.dataset), train_acc, test_acc, time.time() - t_start))
            
            self.scheduler.step()