"""
utils/image_utils.py
Utility functions for loading and saving images for neural style transfer.
"""

from PIL import Image
from torchvision import transforms
import torch

def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = max(image.size)
    if size > max_size:
        size = max_size
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = in_transform(image).unsqueeze(0)
    return image

def save_image(tensor, path):
    image = tensor.to('cpu').clone().detach()
    image = image.squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)
