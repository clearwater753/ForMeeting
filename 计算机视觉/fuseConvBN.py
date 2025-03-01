import torch
import torch.nn as nn

def fuse_conv_bn(conv, bn):

    std = (bn.running_var + bn.eps).sqrt()
    bias = bn.bias - bn.running_mean * bn.weight / std

    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    weights = conv.weight * t

    conv.weight = torch.nn.Parameter(weights)
    conv.bias = torch.nn.Parameter(bias)
    return conv
 
B = nn.BatchNorm2d(64)
print(B.running_mean.shape)