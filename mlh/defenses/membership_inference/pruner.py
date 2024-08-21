import torch
import torch.nn as nn
import torch_pruning as tp
                
class GradGapPruner(tp.pruner.MetaPruner):
    def compute_grad_gap(self, member_grads, nonmember_grads):
        grad_gaps = {}
        for name, grad in member_grads.items():
            grad_gap = torch.abs(grad - nonmember_grads[name])
            grad_gaps[name] = grad_gap
        return grad_gaps
    
    def regularize(self, model, grad_gaps, reg_weight, adaptive_strength=4):
        for m in model.modules():
            # 只对BN层进行处理
            # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
                # 计算与梯度差异相关的自适应正则化项
                # print(grad_gaps[m.weight] / grad_gaps[m.weight].mean())
                adaptive_reg = reg_weight * (grad_gaps[m.weight] / grad_gaps[m.weight].mean()) ** adaptive_strength
                # adaptive_reg = reg_weight

                # 更新梯度，加上自适应正则化
                m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))


class MIAImportance(tp.importance.GroupNormImportance):
    # def __call__(self, group, **kwargs):
    #     # 1. 首先定义一个列表用于存储分组内每一层的重要性
    #     group_imp = [] # (num_bns, num_channels) 
    #     # 2. 迭代分组内的各个层，对BN层计算重要性
    #     for dep, idxs in group: # idxs是一个包含所有可剪枝索引的列表，用于处理DenseNet中的局部耦合的情况
    #         layer = dep.target.module # 获取 nn.Module
    #         prune_fn = dep.handler    # 获取 剪枝函数
    #         # 3. 对每个BN层计算重要性
    #         if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
    #         # if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
    #             local_imp = torch.abs(layer.weight.data) # 计算scale参数的绝对值大小
    #             group_imp.append(local_imp) # 将其保存在列表中
    #     if len(group_imp)==0: return None # 跳过不包含BN层的分组
    #     # 4. 按通道计算平均重要性
    #     group_imp = torch.stack(group_imp, dim=0).mean(dim=0) 
    #     return group_imp # (num_channels, )
    
    def __init__(self, group_reduction='mean', normalizer='mean', grad_gaps=None):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer, bias=False, target_types=(nn.modules.batchnorm._BatchNorm,))
        self.grad_gaps=grad_gaps
    
    @torch.no_grad()
    def __call__(self, group: tp.Group):
        group_imp = []
        group_idxs = []
        
        # 遍历每个剪枝组，并计算该组的重要性
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue

            # 处理不同的剪枝函数
            if prune_fn in [
                tp.function.prune_conv_out_channels,
                tp.function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)

                # 引入 grad gap 进行自适应调整
                if layer.weight in self.grad_gaps:
                    local_imp *= self.grad_gaps[layer.weight][idxs] / self.grad_gaps[layer.weight].mean()

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias.data[idxs].abs().pow(self.p)
                    
                    # 对 bias 也应用 grad gap
                    if layer.bias in self.grad_gaps:
                        local_imp *= self.grad_gaps[layer.bias][idxs] / self.grad_gaps[layer.bias].mean()
                    
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            elif prune_fn in [
                tp.function.prune_conv_in_channels,
                tp.function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)

                if prune_fn == tp.function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                
                local_imp = local_imp[idxs]

                # 应用 grad gap
                if layer.weight in self.grad_gaps:
                    local_imp *= self.grad_gaps[layer.weight][idxs] / self.grad_gaps[layer.weight].mean()

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            elif prune_fn == tp.function.prune_batchnorm_out_channels:
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    
                    # 应用 grad gap
                    if layer.weight in self.grad_gaps:
                        local_imp *= self.grad_gaps[layer.weight][idxs] / self.grad_gaps[layer.weight].mean()
                    
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)

                        # 对 bias 也应用 grad gap
                        if layer.bias in self.grad_gaps:
                            local_imp *= self.grad_gaps[layer.bias][idxs] / self.grad_gaps[layer.bias].mean()

                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

            elif prune_fn == tp.function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)

                    # 应用 grad gap
                    if layer.weight in self.grad_gaps:
                        local_imp *= self.grad_gaps[layer.weight][idxs] / self.grad_gaps[layer.weight].mean()

                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)

                        # 对 bias 也应用 grad gap
                        if layer.bias in self.grad_gaps:
                            local_imp *= self.grad_gaps[layer.bias][idxs] / self.grad_gaps[layer.bias].mean()

                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0:  # 跳过不含参数化层的组
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp