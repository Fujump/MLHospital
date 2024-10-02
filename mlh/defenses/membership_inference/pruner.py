import torch
import torch.nn as nn
import torch_pruning as tp
import torch.nn.functional as F
                
def check_nan(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN found in {name}")
        
def softmax_weight(weight):
    """
    对输入的权重张量应用 softmax，并返回一个与输入形状相同的张量。
    
    参数:
    - weight: 要应用 softmax 的权重张量 (torch.Tensor)

    返回:
    - softmax_weight: 应用 softmax 后的权重张量，形状与输入相同 (torch.Tensor)
    """
    # 将权重展平成一维张量
    flat_weight = weight.view(-1)
    
    # 对展平后的权重应用 softmax
    softmax_flat_weight = F.softmax(flat_weight, dim=0)
    
    # 将 softmax 后的权重重塑回原始形状
    softmax_weight = softmax_flat_weight.view_as(weight)
    
    return softmax_weight
        
class GradGapPruner(tp.pruner.MetaPruner):
    def compute_grad_gap(self, member_grads, nonmember_grads):
        grad_gaps = {}
        for name, grad in member_grads.items():
            grad_gap = torch.abs(grad - nonmember_grads[name])
            # grad_gap = (grad - nonmember_grads[name])
            grad_gaps[name] = grad_gap
        return grad_gaps
    
    def regularize(self, model, grad_gaps, reg_weight, adaptive_strength=5,args=None, **kwargs):
        save_reg=False
        # if (kwargs['e']%5==1) and (kwargs['batch_n']%10==1):
        if (kwargs['batch_n']%100==1):
            save_reg=True
            e, batch_n=kwargs['e'], kwargs['batch_n']
            adaptive_regs={}
        # for name, m in model.named_modules():
        #     # 只对BN层进行处理
        #     # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
        #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
        #         # 计算与梯度差异相关的自适应正则化项
        #         # print(grad_gaps[m.weight] / grad_gaps[m.weight].mean())
        #         adaptive_reg = reg_weight * (grad_gaps[name] / grad_gaps[name].mean()) ** adaptive_strength
        #         # adaptive_reg = reg_weight

        #         # 更新梯度，加上自适应正则化
        #         m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))
        
        # L1正则+clamp
        if args.reg_norm=="l1":
            # print("l1")
            for name, m in model.named_modules():
                # 只对BN层进行处理
                # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
                    # 计算与梯度差异相关的自适应正则化项
                    # print(grad_gaps[m.weight] / grad_gaps[m.weight].mean())
                    max_reg_factor = args.reg_clamp  # 可调节的上界
                    min_value = torch.finfo(torch.float32).eps  # float32 最小正数
                    mean_value = max(grad_gaps[name].mean(), min_value)
                    adaptive_reg = reg_weight * torch.clamp(
                        (grad_gaps[name] / mean_value) ** adaptive_strength,
                        min=0, max=max_reg_factor
                    )
                    # adaptive_reg = reg_weight
                    check_nan(adaptive_reg, "adaptive_reg")

                    # 更新梯度，加上自适应正则化
                    # m.weight.grad.data.add_(adaptive_reg *2*m.weight.data)
                    m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))
                    check_nan(m.weight.grad.data, "Model Weight Gradient")
                    check_nan(m.weight.data, "Model Weight")
                    
                    if save_reg:
                        adaptive_regs[name]=adaptive_reg
            
            # for name, m in model.named_modules():
            #     # 只对BN层进行处理
            #     # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
            #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
            #         # 计算与梯度差异相关的自适应正则化项
            #         # print(grad_gaps[m.weight] / grad_gaps[m.weight].mean())
            #         max_reg_factor = args.reg_clamp  # 可调节的上界
            #         #################
            #         softmax_grad_gap = softmax_weight(grad_gaps[name])
            #         check_nan(softmax_grad_gap, "softmax_grad_gap")
            #         min_value = torch.finfo(torch.float32).eps  # float32 最小正数
            #         mean_value = max(softmax_grad_gap.mean(), min_value)
            #         # print(torch.max(softmax_grad_gap / mean_value))
            #         # check_nan((softmax_grad_gap / mean_value) ** adaptive_strength, "divide")
            #         adaptive_reg = reg_weight * torch.clamp(
            #             (softmax_grad_gap/mean_value) ** adaptive_strength,
            #             min=0, max=max_reg_factor
            #         )
            #         #################
            #         # # softmax_grad_gap = (grad_gaps[name])
            #         # min_value = torch.finfo(torch.float32).eps  # float32 最小正数
            #         # mean_value = max(grad_gaps[name].mean(), min_value)
            #         # adaptive_reg = reg_weight * torch.clamp(
            #         #     torch.exp((grad_gaps[name] / mean_value) * adaptive_strength),
            #         #     min=0, max=max_reg_factor
            #         # )
            #         #################
            #         check_nan(adaptive_reg, "adaptive_reg")

            #         # 更新梯度，加上自适应正则化
            #         # m.weight.grad.data.add_(adaptive_reg *2*m.weight.data)
            #         m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))
            #         check_nan(m.weight.grad.data, "Model Weight Gradient")
            #         check_nan(m.weight.data, "Model Weight")
                    
            #         if save_reg:
            #             adaptive_regs[name]=adaptive_reg
                    
        # L1正则+clamp
        elif args.reg_norm=="l2":
            for name, m in model.named_modules():
                # 只对BN层进行处理
                # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
                    # 计算与梯度差异相关的自适应正则化项
                    # print(grad_gaps[m.weight] / grad_gaps[m.weight].mean())
                    max_reg_factor = args.reg_clamp  # 可调节的上界
                    min_value = torch.finfo(torch.float32).eps  # float32 最小正数
                    mean_value = max(grad_gaps[name].mean(), min_value)
                    adaptive_reg = reg_weight * torch.clamp(
                        (grad_gaps[name] / mean_value) ** adaptive_strength,
                        min=0, max=max_reg_factor
                    )
                    
                    check_nan(adaptive_reg, "adaptive_reg")
                    # adaptive_reg = reg_weight
                    # print(f"reg_weight:{reg_weight}")

                    # 更新梯度，加上自适应正则化
                    m.weight.grad.data.add_(adaptive_reg *2*m.weight.data)
                    
                    check_nan(m.weight.grad.data, "Model Weight Gradient")
                    check_nan(m.weight.data, "Model Weight Gradient")
                    # # 如果使用带动量的优化器，需要对动量缓存进行裁剪
                    # if 'momentum_buffer' in args.optimizer.state[m.weight]:
                    #     # if save_reg:
                    #     #     print('clamping momentum_buffer')
                    #     momentum_buffer = args.optimizer.state[m.weight]['momentum_buffer']
                    #     # 裁剪动量缓存，防止动量累积导致溢出
                    #     momentum_buffer.clamp_(-max_value, max_value)
                    
                    if save_reg:
                        adaptive_regs[name]=adaptive_reg
        
        # L1正则+sigmoid
        # for name, m in model.named_modules():
        #     # 只对BN层进行处理
        #     # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
        #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
        #         # 计算与梯度差异相关的自适应正则化项
        #         # print(grad_gaps[m.weight] / grad_gaps[m.weight].mean())
        #         max_reg_value = args.reg_clamp  # 正则强度的最大值
        #         scale_factor = 0.0001  # 控制sigmoid的缩放力度，调整强弱

        #         # 使用 sigmoid 变换来控制 reg_weight 的范围
        #         adaptive_reg = reg_weight * max_reg_value * torch.sigmoid(
        #             scale_factor * (grad_gaps[name] / grad_gaps[name].mean()) ** adaptive_strength
        #         )
        #         # adaptive_reg = reg_weight

        #         # 更新梯度，加上自适应正则化
        #         # m.weight.grad.data.add_(adaptive_reg *2*m.weight.data)
        #         m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))
        
        # # 设置一个adaptive_reg上界
        # for name, m in model.named_modules():
        #     # 只对BN层进行处理
        #     # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
        #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
                # # 计算与梯度差异相关的自适应正则化项
                # # print(grad_gaps[m.weight] / grad_gaps[m.weight].mean())
                # max_reg_factor = 100000  # 可调节的上界
                # adaptive_reg = reg_weight * torch.clamp(
                #     (grad_gaps[name] / grad_gaps[name].mean()) ** adaptive_strength,
                #     min=0, max=max_reg_factor
                # )
        #         # adaptive_reg = reg_weight

        #         # 更新梯度，加上自适应正则化
        #         m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))
                
                
        # # 对 gap 大于阈值的参数，直接将其梯度和权值设为 0
        # for name, m in model.named_modules():
        #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
        #         # 获取当前模块的 grad gap
        #         gap = grad_gaps[name]
                
        #         # Create a mask where elements with gap > threshold are set to True
        #         mask = gap > gap.mean()*10
                
        #         # Set the gradient to 0 where the gap is greater than the threshold
        #         m.weight.data[mask] = 0
                
        #         # For the remaining elements, apply the adaptive regularization
        #         adaptive_reg = reg_weight * (gap / gap.mean()) ** adaptive_strength
        #         m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))
        #         m.weight.grad.data[mask] = 0
                
        
        # # 只处理包含 'layer3&4' 的模块
        # for name, m in model.named_modules():
        #     # 只处理包含 'layer4' 的模块
        #     if ('layer3' in name) or ('layer4' in name) and isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
        #         # 计算与梯度差异相关的自适应正则化项
        #         adaptive_reg = reg_weight * (grad_gaps[name] / grad_gaps[name].mean()) ** adaptive_strength

        #         # 更新梯度，加上自适应正则化
        #         m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))
        #     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
        #         # 计算与梯度差异相关的自适应正则化项
        #         adaptive_reg = reg_weight 

        #         # 更新梯度，加上自适应正则化
        #         m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))


                    
        if save_reg:
            torch.save(adaptive_regs, f'/data/home/huq/MLHospital/log_distribution/adaptive_regs/adaptive_regs_{e}_{batch_n}.pth')

        
        # # 遍历所有参数及其名称
        # for name, param in model.named_parameters():
        #     # 检查参数的名称是否在 grad_gaps 字典中
        #     if name in grad_gaps:
        #         # 获取参数的模块
        #         # name 的格式为 <module_name>.<parameter_type>
        #         module_name = name.split('.')[0]
        #         module = getattr(model, module_name, None)
                
        #         # 确保模块的类型匹配并且模块存在
        #         if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Conv2d, nn.Linear)):
        #             # 计算与梯度差异相关的自适应正则化项
        #             adaptive_reg = reg_weight * (grad_gaps[name] / grad_gaps[name].mean()) ** adaptive_strength
                    
        #             # 更新梯度，加上自适应正则化
        #             param.grad.data.add_(adaptive_reg * torch.sign(param.data))


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