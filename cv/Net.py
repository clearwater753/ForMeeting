import torch
import torch.nn as nn

class CBL(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=k, stride=2, padding=(k-1)//2)
        self.bn = nn.BatchNorm2d(10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

net = CBL()
out = net(torch.randn(1, 3, 224, 224))
print(out.shape)