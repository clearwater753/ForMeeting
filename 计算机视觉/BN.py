# 手动实现nn.BatchNorm2d

import torch
import torch.nn as nn
class BN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()
        
    def forward(self, x):
        if self.training:
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.var(x, dim=[0, 2, 3], unbiased = False, keepdim=True)
            self._running_mean = self.running_mean.reshape(-1, self.num_features, 1, 1)
            self._running_var = self.running_var.reshape(-1, self.num_features, 1, 1)
            self.running_mean = (1 - self.momentum) * self._running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self._running_var + self.momentum * var
        else:
            mean = self.running_mean.reshape(-1, self.num_features, 1, 1)
            var = self.running_var.reshape(-1, self.num_features, 1, 1)
        x = (x - mean) / (var + self.eps).sqrt()
        self._weight = self.weight.reshape(-1, self.num_features, 1, 1)
        self._bias = self.bias.reshape(-1, self.num_features, 1, 1)
        x = x * self._weight + self._bias
        return x

B = BN(64)
print(B.running_mean.shape)