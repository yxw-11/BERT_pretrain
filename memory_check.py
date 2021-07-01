import os
import json

## External Libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16
from torchvision.datasets import FashionMNIST

train_batch_size = 32
test_batch_size = 32

## Specify Image Transforms
img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

## Download Datasets
train_data = FashionMNIST('./data', transform=img_transform, download=True, train=True)
test_data = FashionMNIST('./data', transform=img_transform, download=True, train=False)

## Initialize Dataloaders
training_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

model_vgg_pre = vgg16(pretrained=True)
# Freeze model weights
for param in model_vgg_pre.parameters():
    param.requires_grad = False
model_vgg_pre.classifier[-1] = nn.Linear(4096, 10)
# Randomly initialize all parameters
model_vgg_pre = model_vgg_pre.cuda()