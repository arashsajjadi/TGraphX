# File: models/pre_encoder.py
import logging
import torch.nn as nn
import torchvision.models as models

_log = logging.getLogger(__name__)

class PreEncoder(nn.Module):
    """
    A pre-encoder module that pre-processes raw node image patches using a ResNet-like architecture.
    If `use_pretrained` is True, it loads a pretrained ResNet-18 (using the weights API).
    Otherwise, it builds a custom ResNet-like module.
    """
    def __init__(self, in_channels, out_channels, use_pretrained=False, custom_params=None):
        super().__init__()
        self.out_channels = out_channels  # Save the output channel count.
        if use_pretrained:
            _log.info("Loading pretrained ResNet-18 weights.")
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Remove the fully-connected layer and global pooling.
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.conv1x1 = nn.Conv2d(512, out_channels, kernel_size=1)

            _log.info("Pretrained ResNet-18 loaded successfully.")
        else:
            # Build a simple custom pre-encoder (example with 2 conv layers)
            hidden_channels = custom_params.get("hidden_channels", out_channels) if custom_params else out_channels
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
            self.conv1x1 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1x1(x)
        return x
