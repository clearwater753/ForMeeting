import torch

class Softmax:
    def __init__(self, dim=-1):
        self.dim = dim
        self.output = None
    
    def forward(self, x):
        # x: (N, class)
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0]
        x = x - x_max
        x = torch.exp(x)
        self.output = x/(torch.sum(x, dim=self.dim, keepdim=True))
        return self.output
    
    def backward(self, grad_input):
        
        y = self.output

        grad_output = y*(grad_input - torch.sum(y*grad_input, dim=self.dim, keepdim=True))
        return grad_output

x = torch.tensor([[1, 2, 3]], requires_grad=False)
grad_input = torch.tensor([[0.1, 0.2, 0.3]])
softmax = Softmax()

output = softmax.forward(x)
grad_output = softmax.backward(grad_input)
print(grad_output.shape)

