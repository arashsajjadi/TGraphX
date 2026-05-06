# File: models/pre_encoder.py
import torch.nn as nn
import torchvision.models as models

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
            print("Loading pretrained ResNet-18 weights...")
            # Use the new weights API from torchvision (>=0.13)
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

            # Check if the weights are successfully loaded by printing the model's parameters
            print("ResNet-18 model loaded. Checking model parameters:")
            for name, param in resnet.named_parameters():
                print(f"{name}: {param.shape}")

            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Remove the fully-connected layer and pooling
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.conv1x1 = nn.Conv2d(512, out_channels, kernel_size=1)

            print("Pretrained weights loaded successfully.")
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
