import torch
import torch.nn as nn
import torch_pruning as tp

class AdaptiveContrastivePruner(tp.pruner.MetaPruner):
    def regularize(self, model, conf_member, conf_nonmember, reg_weight, margin=0.5):
        # 计算member和non-member之间的差距
        confidence_gap = conf_member - conf_nonmember
        
        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        for m in model.modules():  # 遍历所有层
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True:
                # 基于confidence_gap调整正则化强度
                adaptive_reg = reg_weight * torch.relu(confidence_gap + margin)
                
                # 对参数进行自适应正则化
                # param.grad.data.add_(adaptive_reg.mean() * torch.sign(param.data))
                m.weight.grad.data.add_(adaptive_reg.mean() * torch.sign(m.weight.data))
                
class GradGapPruner(tp.pruner.MetaPruner):
    def compute_grad_gap(self, member_grads, nonmember_grads):
        grad_gaps = {}
        for name, grad in member_grads.items():
            grad_gap = torch.abs(grad - nonmember_grads[name])
            grad_gaps[name] = grad_gap
        return grad_gaps
    
    def regularize(self, model, grad_gaps, reg_weight):
        for m in model.modules():
            # 只对BN层进行处理
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
                # 计算与梯度差异相关的自适应正则化项
                adaptive_reg = reg_weight * grad_gaps[m.weight] / grad_gaps[m.weight].mean()

                # 更新梯度，加上自适应正则化
                m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))


class MIAImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        # 1. 首先定义一个列表用于存储分组内每一层的重要性
        group_imp = [] # (num_bns, num_channels) 
        # 2. 迭代分组内的各个层，对BN层计算重要性
        for dep, idxs in group: # idxs是一个包含所有可剪枝索引的列表，用于处理DenseNet中的局部耦合的情况
            layer = dep.target.module # 获取 nn.Module
            prune_fn = dep.handler    # 获取 剪枝函数
            # 3. 对每个BN层计算重要性
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
                local_imp = torch.abs(layer.weight.data) # 计算scale参数的绝对值大小
                group_imp.append(local_imp) # 将其保存在列表中
        if len(group_imp)==0: return None # 跳过不包含BN层的分组
        # 4. 按通道计算平均重要性
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0) 
        return group_imp # (num_channels, )