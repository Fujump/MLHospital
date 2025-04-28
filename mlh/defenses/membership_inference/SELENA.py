import torch
import copy
import numpy as np
import random
import os
import time
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from mlh.defenses.membership_inference.trainer import Trainer
from runx.logx import logx

# -------------------- SELENA 核心函数 -------------------- #
def lr_update(step: int, total_epoch: int, train_size: int, initial_lr: float) -> float:
    """Cosine decay schedule adapted for SELENA training."""
    progress = step / (total_epoch * train_size)
    lr = initial_lr * np.cos(progress * (7 * np.pi) / (2 * 8))
    lr *= np.clip(progress * 100, 0, 1)
    return lr

def get_optimizer(model: torch.nn.Module, configs: dict) -> torch.optim.Optimizer:
    """选择合适的优化器"""
    optimizer_name = configs.get("optimizer", "SGD")
    lr = configs.get("learning_rate", 0.01)
    wd = configs.get("weight_decay", 0.0)
    momentum = configs.get("momentum", 0.9)
    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def inference(model: nn.Module, loader: DataLoader, device: str) -> tuple:
    """评估模型性能，返回loss与正确率"""
    model.eval().to(device)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            correct += (output.argmax(dim=1) == target).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def prepare_teacher_models(
    base_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    configs: dict,
    logger,
):
    """
    根据教师模型数量及排除策略生成多个教师模型，每个教师只使用部分数据进行训练。
    """
    device = configs.get("device", "cpu")
    logger.msg(f"Device: {device}")
    num_teachers = configs.get("num_teachers", 25)
    L = configs.get("num_excluded_teachers_per_sample", 10)
    teacher_models = []
    
    dataset = train_loader.dataset
    data_len = len(dataset)
    
    # 为每个样本随机选择L个不参与训练的教师，其余教师会看到该样本
    teacher_sample_sets = [set() for _ in range(num_teachers)]
    random.seed(0)
    for sample_idx in range(data_len):
        excluded_teachers = random.sample(range(num_teachers), min(L, num_teachers))
        for t_id in range(num_teachers):
            if t_id not in excluded_teachers:
                teacher_sample_sets[t_id].add(sample_idx)

    logger.msg("Overlap-based slicing: each teacher may see a different subset of samples.")

    for t_id in range(num_teachers):
        logger.msg(f"Training teacher {t_id + 1}/{num_teachers}")
        subset_indices = list(teacher_sample_sets[t_id])
        teacher_dataset = Subset(dataset, subset_indices)
        teacher_train_loader = DataLoader(
            teacher_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
        )
        logger.msg(f"Teacher {t_id + 1}/{num_teachers} | Training data: {len(teacher_dataset)} samples")
        teacher = copy.deepcopy(base_model).to(device)
        optimizer = get_optimizer(teacher, configs)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=configs.get("teacher_epochs", 100),
        )
        loss_fn = nn.CrossEntropyLoss()
        epochs = configs.get("teacher_epochs", 100)
        
        for e in range(epochs):
            teacher.train()
            for data, target in teacher_train_loader:
                data, target = data.to(device), target.to(device).long()
                optimizer.zero_grad()
                output = teacher(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

        if test_loader:
            _val_loss, _val_acc = inference(teacher, test_loader, device)
            logger.msg(f"Test Loss: {_val_loss:.4f} | Test Acc: {_val_acc:.4f}")

        teacher_models.append(teacher)

    return teacher_models, teacher_sample_sets

def train_SELENA(model: nn.Module, train_loader: DataLoader,
          test_loader: DataLoader, configs: dict, logger) -> nn.Module:
    """
    SELENA的训练流程：
    1. 训练多个教师模型
    2. 利用教师模型对训练集样本进行软标签预测，构造新的数据集
    3. 使用KL散度损失对模型进行训练，并按cosine学习率调度更新
    """
    teacher_models, teacher_sample_sets = prepare_teacher_models(model, train_loader, test_loader, configs, logger)
    device = configs.get("device", "cpu")
    model = model.to(device)
    epochs = configs.get("epochs", 100)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)
    schedule = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs
    )
    # schedule = lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: lr_update(step, epochs, len(train_loader), configs.get("learning_rate", 0.1)),
    # )
    
    # 根据教师模型生成软标签数据集
    new_data, new_labels = [], []
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        preds = []
        for t_id, t_model in enumerate(teacher_models):
            # 如果该教师模型未见过该样本，则使用其预测
            if idx not in teacher_sample_sets[t_id]:
                with torch.no_grad():
                    t_output = t_model(data)
                    preds.append(torch.softmax(t_output, dim=1))
        if preds:
            avg_pred = torch.mean(torch.stack(preds, dim=0), dim=0)
            new_data.append(data.cpu())
            new_labels.append(avg_pred.cpu())
    new_data = torch.cat(new_data) if new_data else torch.empty(0)
    new_labels = torch.cat(new_labels) if new_labels else torch.empty(0)

    selena_dataset = torch.utils.data.TensorDataset(new_data, new_labels)
    selena_loader = torch.utils.data.DataLoader(selena_dataset, batch_size=train_loader.batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for data, soft_target in selena_loader:
            data, soft_target = data.to(device), soft_target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(output), soft_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (output.argmax(dim=1) == soft_target.argmax(dim=1)).sum().item()
        schedule.step()

        logger.msg(f"Epoch [{epoch+1}/{epochs}] | SELENA Loss: {total_loss/len(selena_loader):.4f} | SELENA Acc: {correct/len(selena_dataset):.4f}")

        if test_loader:
            val_loss, val_acc = inference(model, test_loader, device)
            logger.msg(f"Test Loss: {val_loss:.4f} | Test Acc: {val_acc:.4f}")

    model.to("cpu")
    return model

# -------------------- 封装为 TrainTargetSELENA -------------------- #
class TrainTargetSELENA(Trainer):
    def __init__(self, model, device="cuda", epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=0.0,
                 teacher_epochs=100, num_teachers=25, num_excluded_teachers_per_sample=10, log_path="./"):
        """
        参数说明：
         - model: 需要训练的模型
         - device: 训练设备
         - epochs: 训练轮数
         - learning_rate, momentum, weight_decay: 优化器参数
         - teacher_epochs: 每个教师模型训练的轮数
         - num_teachers: 教师模型数量
         - num_excluded_teachers_per_sample: 每个样本排除的教师数量
         - log_path: 日志保存路径
        """
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.teacher_epochs = teacher_epochs
        self.num_teachers = num_teachers
        self.num_excluded_teachers_per_sample = num_excluded_teachers_per_sample
        self.log_path = log_path
        logx.initialize(logdir=self.log_path, coolname=False, tensorboard=False)

    def eval(self, data_loader):
        """
        使用普通的交叉熵损失计算测试集上的准确率，并返回最终准确率（百分比）。
        """
        self.model.eval().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device).long()
                outputs = self.model(data)
                loss = loss_fn(outputs, target)
                total_loss += loss.item()
                correct += (outputs.argmax(dim=1) == target).sum().item()
        final_acc = 100.0 * correct / len(data_loader.dataset)
        return final_acc

    def train(self, train_loader, test_loader):
        """
        按照SELENA的训练流程进行训练。训练完成后会在日志中输出最终测试准确率。
        """
        configs = {
            "device": self.device,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "teacher_epochs": self.teacher_epochs,
            "num_teachers": self.num_teachers,
            "num_excluded_teachers_per_sample": self.num_excluded_teachers_per_sample,
        }
        logger = logx
        t_start = time.time()
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # 调用SELENA的训练流程
        self.model = train_SELENA(self.model, train_loader, test_loader, configs, logger)
        final_acc = self.eval(test_loader)
        logger.msg(f"Final Test Accuracy: {final_acc:.2f}% in {time.time()-t_start:.2f} seconds")
