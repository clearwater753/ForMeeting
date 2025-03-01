import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x)) # 防止溢出
    return exp_x / exp_x.sum(axis=0)

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def silu(x):
    return x * sigmoid(x)