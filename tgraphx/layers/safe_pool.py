# File: layers/safe_pool.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeMaxPool2d(nn.Module):
    """
    A MaxPool2d module that checks whether the input spatial dimensions are large enough
    for the specified kernel size. If not, it returns the input unchanged.
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.ceil_mode = ceil_mode

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        # Only apply pooling if both dimensions are at least as large as the kernel.
        if h < self.kernel_size[0] or w < self.kernel_size[1]:
            return x
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode)
