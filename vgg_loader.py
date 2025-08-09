"""
models/vgg_loader.py
Utility to load a pre-trained VGG19 model for style transfer.
"""

import torch
from torchvision import models

def get_vgg19(device):
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg
