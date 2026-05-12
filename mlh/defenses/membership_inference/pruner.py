import torch
import torch.nn as nn
# import torch_pruning as tp
import torch.nn.functional as F
import os
                
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
        
class PAST:
    def compute_grad_gap(self, member_grads, nonmember_grads):
        grad_gaps = {}
        for name, grad in member_grads.items():
            grad_gap = torch.abs(grad - nonmember_grads[name])
            grad_gaps[name] = grad_gap
        return grad_gaps

    def regularize(self, model, grad_gaps, reg_weight, adaptive_strength=5, args=None, **kwargs):
        save_reg = False
        # if (kwargs['e']%5==1) and (kwargs['batch_n']%10==1):
        if (kwargs['batch_n'] % 100 == 1):
            save_reg = True
            e, batch_n = kwargs['e'], kwargs['batch_n']
            adaptive_regs = {}

        # L1正则+clamp
        if args.reg_norm == "l1":
            print("l1 regularization with clamp with parameter reg_weight, adaptive_strength:", reg_weight, adaptive_strength)
            for name, m in model.named_modules():
                # 只对BN层进行处理
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
                    m.weight.grad.data.add_(adaptive_reg * torch.sign(m.weight.data))
                    check_nan(m.weight.grad.data, "Model Weight Gradient")
                    check_nan(m.weight.data, "Model Weight")

                    if save_reg:
                        adaptive_regs[name] = adaptive_reg

        # L2正则+clamp
        elif args.reg_norm == "l2":
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
                    m.weight.grad.data.add_(adaptive_reg * 2 * m.weight.data)

                    check_nan(m.weight.grad.data, "Model Weight Gradient")
                    check_nan(m.weight.data, "Model Weight")
                    if save_reg:
                        adaptive_regs[name] = adaptive_reg

        if save_reg:
            path = f'/data/home/zhanghx/MLHospital/past/adaptive_regs_{e}_{batch_n}.pth'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(adaptive_regs, path)