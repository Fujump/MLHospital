import copy
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from runx.logx import logx
from mlh.defenses.membership_inference.trainer import Trainer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TrainTargetMIST(Trainer):
    def __init__(self, model, device="cuda", epochs=100, optimizer="SGD",
                 learning_rate=0.01, weight_decay=5e-4, momentum=0.9,
                 num_submodels=2, xdiff_lambda=1, log_path="./"):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.num_submodels = num_submodels
        self.xdiff_lambda = xdiff_lambda

        # 存储配置信息，方便后续获取
        self.configs = {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "epochs": epochs,
            "device": device,
            "num_submodels": num_submodels,
            "xdiff_lambda": xdiff_lambda
        }

        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()
        

        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        logx.initialize(logdir=self.log_path, coolname=False, tensorboard=False)

    @staticmethod
    def get_optimizer(model, configs):
        optimizer_name = configs.get("optimizer", "SGD")
        learning_rate = configs.get("learning_rate", 0.001)
        weight_decay = configs.get("weight_decay", 0.0)
        momentum = configs.get("momentum", 0.0)
        if optimizer_name == "SGD":
            return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "Adam":
            return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented.")

    def eval(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                predicted = torch.argmax(outputs, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        final_acc = 100 * correct / total
        return final_acc

    def train(self, train_loader, test_loader):
        print(f"training with MIST, lambda = {self.xdiff_lambda}")
        total_samples = len(train_loader.dataset)
        indices = list(range(total_samples))
        subset_size = total_samples // self.num_submodels
        subsets_indices = [indices[i * subset_size: (i + 1) * subset_size] for i in range(self.num_submodels)]
        if total_samples % self.num_submodels != 0:
            subsets_indices[-1].extend(indices[self.num_submodels * subset_size:])

        # 构造每个子模型对应的 DataLoader，保证与 train_loader 参数一致
        subset_loaders = []
        for idx_list in subsets_indices:
            subset = Subset(train_loader.dataset, idx_list)
            loader = DataLoader(subset, batch_size=train_loader.batch_size,
                                shuffle=True, num_workers=train_loader.num_workers)
            subset_loaders.append(loader)

        # **提前创建子模型和优化器**
        avg_state_dict = self.model.state_dict()
        submodels = [copy.deepcopy(self.model).to(self.device) for _ in range(self.num_submodels)]
        sub_optimizers = [self.get_optimizer(submodels[i], self.configs) for i in range(self.num_submodels)]
        sub_schedulers = [optim.lr_scheduler.CosineAnnealingLR(sub_optimizers[i], T_max=self.epochs) for i in range(self.num_submodels)]


        t_start = time.time()
        for epoch in range(1, self.epochs + 1):
            logx.msg(f"Epoch {epoch}/{self.epochs} training with MIST...")

            # # --- 第一步：在各子集上分别进行常规训练，得到子模型 ---
            # submodels = []
            # sub_optimizers = []
            # for _ in range(self.num_submodels):
            #     submodel = copy.deepcopy(self.model).to(self.device)
            #     submodels.append(submodel)
            #     sub_optimizers.append(self.get_optimizer(submodel, self.configs))
            # 在各子集上各自训练一个epoch
            for submodel in submodels:
                submodel.load_state_dict(avg_state_dict)
            for submodel, sub_optimizer, loader in zip(submodels, sub_optimizers, subset_loaders):
                submodel.train()
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device).long()
                    sub_optimizer.zero_grad()
                    outputs = submodel(data)
                    loss = self.criterion(outputs, target)
                    loss.backward()
                    sub_optimizer.step()

            # --- 第二步：对子模型进行“反事实”更新，最小化与其他子模型预测均值的均方误差 ---
            for idx, (submodel, sub_optimizer, loader) in enumerate(zip(submodels, sub_optimizers, subset_loaders)):
                submodel.train()
                for data, _ in loader:
                    data = data.to(self.device)
                    output_i = submodel(data)
                    prob_i = torch.softmax(output_i, dim=1)
                    other_probs = 0
                    count = 0
                    for j, other_model in enumerate(submodels):
                        if j == idx:
                            continue
                        other_model.eval()
                        with torch.no_grad():
                            output_j = other_model(data)
                            other_probs += torch.softmax(output_j, dim=1)
                        count += 1
                    avg_other = other_probs / count if count > 0 else prob_i
                    loss_mse = self.xdiff_lambda * self.mse_loss_fn(prob_i, avg_other)
                    sub_optimizer.zero_grad()
                    loss_mse.backward()
                    sub_optimizer.step()
                sub_schedulers[idx].step()
            # --- 第三步：对子模型参数进行平均，更新到主模型 ---
            with torch.no_grad():
                submodels_state = [submodel.state_dict() for submodel in submodels]
                for key in submodels_state[0]:
                    avg_state_dict[key] = sum(sub_state[key] for sub_state in submodels_state) / len(submodels_state)
                self.model.load_state_dict(avg_state_dict)

            # 每个epoch结束后输出日志，并在测试集上评估
            train_acc = self.eval(train_loader)
            if test_loader is not None:
                test_acc = self.eval(test_loader)
                logx.msg(f"Epoch: {epoch}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, Time Elapsed: {time.time()-t_start:.2f}s")
            else:
                logx.msg(f"Epoch: {epoch}, Train Acc: {train_acc:.3f}, Time Elapsed: {time.time()-t_start:.2f}s")
