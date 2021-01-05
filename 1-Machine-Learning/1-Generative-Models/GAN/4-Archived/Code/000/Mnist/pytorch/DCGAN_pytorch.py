import torch
from torch import nn
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import time

EPOCHS = 5
BATCH_SIZE = 128
image_channel = 1
filters_num = 128
num_square_examples_to_generate = 5
noise_dim = 100
lr = 0.0002
beta1 = 0.5

sample_save_folder = './output_DCGAN'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5,std=0.5)])

data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)


def make_generator_model():

    model = nn.Sequential(
        nn.ConvTranspose2d(in_channels=noise_dim,
                           out_channels=filters_num * 2,
                           kernel_size=7,
                           stride=1,
                           padding=0,
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

    return model


def make_discriminator_model():

    model = nn.Sequential(
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
                  out_channels=1,
                  kernel_size=7,
                  stride=1,
                  padding=0,
                  bias=False)
    )

    return model

generator = make_generator_model().to(device)
discriminator = make_discriminator_model().to(device)


cross_entropy = nn.BCEWithLogitsLoss()

def discriminator_loss_fn(real_output, fake_output): 
    real_loss = cross_entropy(torch.ones_like(real_output, device=device), real_output)
    fake_loss = cross_entropy(torch.zeros_like(fake_output, device=device), fake_output)
    return real_loss + fake_loss

def generator_loss_fn(fake_output):
    return cross_entropy(torch.ones_like(fake_output, device=device), fake_output)
    

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))



def generate_and_save_images(predictions, epoch):

    if not os.path.exists(sample_save_folder):
        os.makedirs(sample_save_folder)

    fig = plt.figure(figsize=(num_square_examples_to_generate,
                              num_square_examples_to_generate))

    for i in range(predictions.shape[0]):
        plt.subplot(num_square_examples_to_generate, num_square_examples_to_generate, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(sample_save_folder + f'/image_at_epoch_{epoch:04d}.png')
    plt.close()



num_examples_to_generate = 5
seed = torch.randn(num_examples_to_generate ** 2, noise_dim, 1, 1, device=device)

for epoch in range(EPOCHS):

    start = time.time()

    print(f'Epoch {epoch + 1}/{EPOCHS}'.center(40,'-'))

    for i, (images, _) in enumerate(data_loader_train):

        noise = torch.randn(BATCH_SIZE, noise_dim, 1, 1, device=device)    

        
        real_images = images.to(device)
        real_output = discriminator(real_images)
        
        
        generated_images = generator(noise)
        fake_output = discriminator(generated_images.detach())

        discriminator_loss = discriminator_loss_fn(real_output, fake_output)

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()


        noise = torch.randn(BATCH_SIZE, noise_dim, 1, 1, device=device)
        generated_images = generator(noise)
        fake_output = discriminator(generated_images)
        generator_loss = generator_loss_fn(fake_output)

        
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        if i % 50 == 0:
            print(f'step = {i}, generator_loss = {generator_loss.item():.3f}, discriminator_loss = {discriminator_loss.item():.3f}')

    # generator.eval()
    # with torch.no_grad():
    #     predictions = generator(seed).detach().cpu()
    # generate_and_save_images(predictions, epoch)

    # print(f'Time: {time.time() - start}')

        






