# numpy手撕卷积

import numpy as np

def conv2d(x, kernel, stride=1, padding=1):
    # x: (B, Cin, H, W)
    # kernel: (Cout, Cin, kH, kW)
    B, Cin, H, W = x.shape
    Cout, _, kH, kW = kernel.shape
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    out = np.zeros((B, Cout, H_out, W_out))
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    for f in range(Cout):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + kH
                w_start = j * stride
                w_end = w_start + kW
                # x_slice: (B, Cin, kH, kW)
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                # kernel[f, :, :, :]: (Cin, kH, kW) -> (1, Cin, kH, kW)
                out[:, f, i, j] = np.sum(x_slice * np.expand_dims(kernel[f, :, :, :], axis=0), axis=(1, 2, 3))
    return out
x = np.random.randn(2, 3, 224, 224)
kernel = np.random.randn(16, 3, 3, 3)
out = conv2d(x, kernel, stride=2)
print(out.shape)