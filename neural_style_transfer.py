
"""
Neural Style Transfer Demo Script (Modular)

Requirements:
- torch
- torchvision
- pillow

Usage:
    python neural_style_transfer.py

Place 'content.jpg' and 'style.jpg' in the 'images/' folder.
The output will be saved in the 'output/' folder as 'output.jpg'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.image_utils import load_image, save_image
from models.vgg_loader import get_vgg19

# --- Main code ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
content_path = os.path.join('images', 'content.jpg')
style_path = os.path.join('images', 'style.jpg')
output_path = os.path.join('output', 'output.jpg')

content = load_image(content_path).to(device)
style = load_image(style_path).to(device)

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content layer
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


# Load pre-trained VGG19 using modular loader
vgg = get_vgg19(device)

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

target = content.clone().requires_grad_(True).to(device)

style_weights = {'conv1_1': 1.0,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}
content_weight = 1e4
style_weight = 1e2

optimizer = optim.Adam([target], lr=0.003)

steps = 200  # Reduce for faster demo

for i in range(1, steps+1):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (target_feature.shape[1]**2)
    total_loss = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(f"Step {i}, Total loss: {total_loss.item():.2f}")

# Save output

# Save output using utility
save_image(target, output_path)
print(f"Style transfer complete! Output saved as '{output_path}'.")
