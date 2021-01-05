import torch
from torch import nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

image_size = 128
batch_size = 256

dataroot = "D:/Data/Face/celeba/raw"
dataset = datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, drop_last=True, num_workers=4)


print(torch.min(dataset[0][0]))

img = dataset[0][0].permute(1,2,0)
img = img * 0.5 + 0.5
plt.imshow(img)
plt.show()