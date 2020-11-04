import torch
from torch import nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

noise_dim = 100
image_channel = 3
filters_num = 32
batch_size = 256
image_size = 64

dataroot = "D:/Data/Face/celeba"
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_generator_model():

    model = nn.Sequential(
        nn.ConvTranspose2d(in_channels=noise_dim,
                           out_channels=filters_num * 8,
                           kernel_size=4,
                           stride=1,
                           padding=0,
                           bias=False),
        nn.BatchNorm2d(filters_num * 8),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(in_channels=filters_num * 8,
                           out_channels=filters_num * 4,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=False),
        nn.BatchNorm2d(filters_num * 4),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(in_channels=filters_num * 4,
                           out_channels=filters_num * 2,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=False),
        nn.BatchNorm2d(filters_num * 2),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(in_channels=filters_num * 2,
                           out_channels=filters_num,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=False),
        nn.BatchNorm2d(filters_num),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(in_channels=filters_num,
                           out_channels=image_channel,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=False),
        nn.Tanh()
    )

    return model


def make_discriminator_model():

    model = nn.Sequential(
        nn.Conv2d(in_channels=image_channel,
                  out_channels=filters_num,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  bias=False),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0.3, inplace=True),

        nn.Conv2d(in_channels=filters_num,
                  out_channels=filters_num * 2,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  bias=False),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0.3, inplace=True),

        nn.Conv2d(in_channels=filters_num * 2,
                  out_channels=filters_num * 4,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  bias=False),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0.3, inplace=True),

        nn.Conv2d(in_channels=filters_num * 4,
                  out_channels=filters_num * 8,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  bias=False),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0.3, inplace=True),

        nn.Conv2d(in_channels=filters_num * 8,
                  out_channels=1,
                  kernel_size=4,
                  stride=1,
                  padding=0,
                  bias=False)
    )

    return model


class Generator(nn.Module):
    def __init__(self):

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_dim,
                               out_channels=filters_num * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(filters_num * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=filters_num * 8,
                               out_channels=filters_num * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(filters_num * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=filters_num * 4,
                               out_channels=filters_num * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(filters_num * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=filters_num * 2,
                               out_channels=filters_num,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(filters_num),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=filters_num,
                               out_channels=image_channel,
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
            nn.Conv2d(in_channels=image_channel,
                      out_channels=filters_num,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=filters_num,
                      out_channels=filters_num * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=filters_num * 2,
                      out_channels=filters_num * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=filters_num * 4,
                      out_channels=filters_num * 8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=filters_num * 8,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False)
        )

    def forward(self, input):
        return self.main(input)

