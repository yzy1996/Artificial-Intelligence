import torch
from torch import nn
from torchvision import transforms,datasets 
import time
import os
import matplotlib.pyplot as plt
import numpy as np

sample_save_folder = './output_DCGAN_2'

noise_dim = 100
image_channel = 3
filters_num = 64
batch_size = 256
image_size = 64

dataroot = "D:/Data/Face/celeba/Male/2"
dataset = datasets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, drop_last=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

ndf = 64
ngf = 64
nz = 100
nc = 3
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

loss = nn.BCELoss()

lr = 0.0002
beta1 = 0.5
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

def generate_and_save_images(predictions, epoch):

    if not os.path.exists(sample_save_folder):
        os.makedirs(sample_save_folder)

    fig = plt.figure(figsize=(5, 5))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        img = predictions[i].permute(1,2,0)
        img = img * 0.5 + 0.5
        plt.imshow(img)
        plt.axis('off')

    plt.savefig(sample_save_folder + f'/image_at_epoch_{epoch+1:02d}.png')
    plt.close()

fixed_noise = torch.randn(25, noise_dim, 1, 1, device=device)
num_epochs = 50
for epoch in range(num_epochs):

    start1 = time.time()
    generator.train()
    for i, (real_images, _) in enumerate(dataloader, 1):

        start2 = time.time()

        ###################
        optimizerD.zero_grad()
        real_images = real_images.to(device)
        real_output = discriminator(real_images)
        label = torch.ones_like(real_output, device=device)
        
        loss_D_real = loss(real_output, label)
        loss_D_real.backward()

        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_image = generator(noise)
        fake_output = discriminator(fake_image.detach())
        label = torch.zeros_like(fake_output, device=device)

        loss_D_fake = loss(fake_output, label)
        loss_D_fake.backward()

        loss_D = loss_D_real + loss_D_fake
        optimizerD.step()
        
        ###################
        optimizerG.zero_grad()
        output = discriminator(fake_image)
        label = torch.ones_like(fake_output, device=device)
        loss_G = loss(output, label)

        loss_G.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f'[{epoch + 1}/{num_epochs}] [{i}/{len(dataloader)}], generator_loss = {loss_G.item():.3f}, discriminator_loss = {loss_D.item():.3f}, time = {time.time() - start2}')

    print(f'time = {time.time() - start1}'.center(60,'-'))

    generator.eval()
    with torch.no_grad():
        predictions = generator(fixed_noise).detach().cpu()
    generate_and_save_images(predictions, epoch)







