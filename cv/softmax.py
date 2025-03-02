import torch

class Softmax:
    def __init__(self, dim=-1):
        self.dim = dim
        self.output = None

    def forward(self, x):
        """
        x: (batch_size, num_classes)
        Softmax 前向传播
        """
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0]
        x_exp = torch.exp(x - x_max)
        self.output = x_exp / torch.sum(x_exp, dim=self.dim, keepdim=True)
        return self.output

    def backward(self, grad_input):
        """
        Softmax 反向传播
        :param grad_output: 损失函数对 Softmax 输出的梯度
        """
        # Softmax 输出
        y = self.output

        # 计算梯度
        grad_output = y * (grad_input - torch.sum(grad_input * y, dim=self.dim, keepdim=True))
        return grad_output

# 测试代码
if __name__ == "__main__":
    # 创建输入张量
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=False)

    # 创建 Softmax 模块
    softmax = Softmax(dim=1)

    # 前向传播
    output = softmax.forward(x)
    print("Softmax Output:", output)

    # 模拟损失函数对 Softmax 输出的梯度
    grad_input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # 反向传播
    grad_output = softmax.backward(grad_input)
    print("Gradient w.r.t. input:", grad_output)