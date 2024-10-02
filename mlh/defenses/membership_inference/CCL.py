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
import torch.optim.lr_scheduler as lr_scheduler

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


class TrainTargetCCL(Trainer):
    def __init__(self, model, device="cuda:0", num_class=10, epochs=100, learning_rate=0.1, momentum=0.9, weight_decay=5e-4, smooth_eps=0.8, alpha=0.5, loss_type="ccel", log_path="./"):

        super().__init__()

        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs
        self.smooth_eps = smooth_eps

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), learning_rate, momentum, weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=self.epochs)
        self.scheduler = get_scheduler(scheduler_name = 'multi_step2', optimizer =self.optimizer, t_max=self.epochs)
        self.loss_type = loss_type
        self.alpha=alpha
        self.criterion = self.initialize_criterion()

        # self.log_path = "%smodel_%s_bs_%s_dataset_%s/%s/label_smoothing_%.1f" % (self.opt.model_save_path, self.opt.model,
        # #                                                                           self.opt.batch_size, self.opt.dataset, self.opt.mode, self.opt.smooth_eps)
        # self.model_save_name = 'model_%s_label_smoothing_%.1f' % (
        #     self.opt.mode, self.opt.smooth_eps)

        # logx.initialize(logdir=self.log_path,
        #                 coolname=False, tensorboard=False)

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

    def initialize_criterion(self):
        """Initialize the loss criterion."""
        return get_loss(loss_type=self.loss_type, device=self.device, num_classes=self.num_class,alpha=self.alpha)


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

    def train(self, train_loader, test_loader):

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
            for img, label in train_loader:
                self.model.zero_grad()
                batch_n += 1

                img, label = img.to(self.device), label.to(self.device)
                # print("img", img.shape)
                logits = self.model(img)
                loss = self.criterion(logits, label)

                loss.backward()
                self.optimizer.step()

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

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, "%s.pth" % self.model_save_name))
        
        
        
def get_loss(loss_type, device, alpha=0.5, beta=0.01, gamma=0, num_classes = 10, reduction = "mean"):
    CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "focal": FocalLoss(gamma = alpha, beta = beta),
        "ccel":CCEL(alpha = alpha, beta = beta),
        "ccql":CCQL(alpha = alpha, beta = beta),
        "focal_ccel": FocalCCEL(alpha = alpha, beta = beta, gamma = gamma),
        }
    return CONFIG[loss_type]


def focal_loss(input_values, gamma, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
class FocalLoss(nn.Module):
    def __init__(self, gamma=0., beta = 1,reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        return self.beta *focal_loss(F.cross_entropy(input, target, reduction="none"), self.gamma, reduction = self.reduction)
    
def taylor_exp(input_values, alpha, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = alpha*input_values - (1-alpha)*(p+torch.pow(p,2)/2)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")

def concave_exp_loss(input_values, gamma =1, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = torch.exp(gamma*p)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
    
def ce_concave_exp_loss(input_values, alpha, beta):
    p = torch.exp(-input_values)
    
    loss = alpha * input_values - beta *torch.exp(p)
    return loss




class CCEL(nn.Module):
    def __init__(self, alpha = 0.5, beta = 0.05, gamma=1.0, tau =1, reduction='mean'):
        super(CCEL, self).__init__()
        assert gamma >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction_ = reduction
    def forward(self, input, target):
        # Calculate the cross-entropy loss without reduction
        ce_loss = F.cross_entropy(input, target, reduction="none")
        # Pass the calculated cross-entropy loss along with other parameters to your custom loss function
        # Ensure that the 'reduction' argument is not passed again if it's already expected by ce_concave_exp_loss function
        modified_loss = ce_concave_exp_loss(ce_loss, self.alpha, (1-self.alpha))
        # Apply the beta scaling and reduce the loss as needed
        if self.reduction_ == 'mean':
            return self.beta * modified_loss.mean()
        elif self.reduction_ == 'sum':
            return self.beta * modified_loss.sum()
        else:
            # If reduction is 'none', just return the scaled loss
            return self.beta * modified_loss


class CCQL(nn.Module):
    def __init__(self, alpha = 1,beta =1,reduction='mean'):
        super(CCQL, self).__init__()
        #assert gamma >= 1e-7
        self.gamma = 2
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        return self.beta*taylor_exp(ce, self.alpha, self.reduction)
    

class FocalCCEL(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0,reduction='mean'):
        super(FocalCCEL, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        losses = focal_loss(ce, gamma = self.gamma, reduction="none")
        cel = concave_exp_loss(ce,reduction="none")
        loss = self.alpha * losses + (1-self.alpha)*cel
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return self.beta*loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
        
####################

def get_scheduler(scheduler_name, optimizer, decay_epochs=1, decay_factor=0.1, t_max=50):
    """
    Get the specified learning rate scheduler instance.

    Parameters:
        scheduler_name (str): The name of the scheduler, can be 'step' or 'cosine'.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        decay_epochs (int): Number of epochs for each decay period, used for StepLR scheduler (default is 1).
        decay_factor (float): The factor by which the learning rate will be reduced after each decay period,
                             used for StepLR scheduler (default is 0.1).
        t_max (int): The number of epochs for the cosine annealing scheduler (default is 50).

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The instance of the selected scheduler.
    
    """
    if isinstance(optimizer, torch.optim.Adam):
        return DummyScheduler()
    if scheduler_name.lower() == 'dummy':
        return DummyScheduler()
    if scheduler_name.lower() == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=decay_factor)
    elif scheduler_name.lower() == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_name.lower() == "multi_step":
        decay_epochs = [150, 225]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step150":
        decay_epochs = [50, 100]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step2":
        decay_epochs = [40, 80]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step_imagenet":
        decay_epochs = [30, 60]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step_wide_resnet":
        decay_epochs = [60,120,160]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.2)
    else:
        raise ValueError("Unsupported scheduler name.")

    return scheduler