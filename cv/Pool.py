import numpy as np

def Pool(x, kernel_size, stride=2, mode='max'):
    B, Cin, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    out = np.zeros((B, Cin, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + kernel_size
            w_start = j * stride
            w_end = w_start + kernel_size
            if mode == 'max':
                out[:, :, i, j] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
            elif mode == 'avg':
                out[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
    
    return out

out = Pool(np.random.randn(2, 3, 224, 224), 2, 2, 'max')
print(out.shape)