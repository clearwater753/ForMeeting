import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 权重因子
        self.gamma = gamma  # 焦点因子
        self.reduction = reduction  # 'mean' 或 'sum'，决定返回的损失形态

    def forward(self, inputs, targets):
        # inputs: [N, C]，N为batch_size，C为类别数
        # targets: [N]，真实标签
        # inputs: 网络输出的预测概率, targets: 真实标签
        # 假设输入是logits，进行softmax操作得到概率
        inputs = F.softmax(inputs, dim=1)
        targets = targets.view(-1, 1)  # 扁平化，确保目标为 (batch_size, 1)
        
        # 获取p_t，目标类别的预测概率
        p_t = inputs.gather(1, targets)
        
        # 计算Focal Loss
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)
        # 如果选择进行平均或求和
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
