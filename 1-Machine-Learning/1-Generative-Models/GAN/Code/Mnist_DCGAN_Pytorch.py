'''
Mnist_DCGAN_Pytorch

by Zhiyuan Yang
'''

import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 定义常量 Constant
EPOCHS = 50
BATCH_SIZE = 128
BUFFER_SIZE = 60000

FILTER_NUM = 128
NOISE_DIM = 100
IMAGE_CHANNEL = 1

NUM_TO_GENERATE = 5  # square


# 创建输出文件夹
OUTPUT_PATH = Path('./Output_Mnist_DCGAN_Pytorch')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=NOISE_DIM,
                               out_channels=FILTER_NUM * 2,
                               kernel_size=7,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(FILTER_NUM * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=FILTER_NUM * 2,
                               out_channels=FILTER_NUM,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(FILTER_NUM),
            nn.LeakyReLU(0.2, inplace=True),


            nn.ConvTranspose2d(in_channels=FILTER_NUM,
                               out_channels=IMAGE_CHANNEL,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=IMAGE_CHANNEL,
                      out_channels=FILTER_NUM,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=FILTER_NUM,
                      out_channels=FILTER_NUM * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=FILTER_NUM * 2,
                      out_channels=1,
                      kernel_size=7,
                      stride=1,
                      padding=0,
                      bias=False)
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

cross_entropy = nn.BCEWithLogitsLoss()


def discriminator_loss_fn(real_output, fake_output):
    real_loss = cross_entropy(real_output, torch.ones_like(real_output, device=device))
    fake_loss = cross_entropy(fake_output, torch.zeros_like(fake_output, device=device))
    return (real_loss + fake_loss) / 2


def generator_loss_fn(fake_output):
    return cross_entropy(fake_output, torch.ones_like(fake_output, device=device))


generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


def generate_and_save_images(predictions, epoch):

    IMAGE_PATH = OUTPUT_PATH / 'image'

    if not IMAGE_PATH.exists():
        IMAGE_PATH.mkdir(parents=True)

    fig = plt.figure(figsize=(NUM_TO_GENERATE, NUM_TO_GENERATE))

    for i in range(predictions.shape[0]):
        plt.subplot(NUM_TO_GENERATE, NUM_TO_GENERATE, i + 1)
        img = predictions[i].permute(1, 2, 0)
        img = img * 0.5 + 0.5
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.savefig(IMAGE_PATH / f'image_at_epoch_{epoch:02d}.png')
    plt.close()


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)


seed = torch.randn(NUM_TO_GENERATE ** 2, NOISE_DIM, 1, 1, device=device)

for epoch in range(EPOCHS):

    start = time.time()

    print(f'Epoch {epoch + 1}/{EPOCHS}'.center(60, '-'))

    for i, (images, _) in enumerate(data_loader_train):

        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1, device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        discriminator_optimizer.zero_grad()

        real_images = images.to(device)
        generated_images = generator(noise)

        real_output = discriminator(real_images)
        fake_output = discriminator(generated_images.detach())

        discriminator_loss = discriminator_loss_fn(real_output, fake_output)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # ---------------------
        #  Train Generator
        # ---------------------

        generator_optimizer.zero_grad()
        fake_output = discriminator(generated_images)
        generator_loss = generator_loss_fn(fake_output)

        generator_loss.backward()
        generator_optimizer.step()

        if i % 50 == 0:
            print(f'step = {i}, generator_loss = {generator_loss.item():.3f}, discriminator_loss = {discriminator_loss.item():.3f}')

    generator.eval()
    with torch.no_grad():
        predictions = generator(seed).detach().cpu()
    generate_and_save_images(predictions, epoch + 1)

    print(f'Time: {time.time() - start}')

torch.save(generator, OUTPUT_PATH / 'Mnist_DCGAN.pkl')