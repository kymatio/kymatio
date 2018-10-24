"""
Description:
This example computes the scattering transform at scale 2 for MNIST digits and trains a convolutional network to invert
the scattering transform. After only two epochs, it produces a network that transforms a linear interpolation in the
scattering space into a nonlinear interpolation in the image space.

Remarks:
This example creates a 'temp' folder to store the dataset, the model after two epochs and the path, which consists of a
sequence of images stored in the folder ./temp/path. Two epochs take roughly 5 minutes in a Quadro M6000.

Reference:
https://arxiv.org/abs/1805.06621
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scattering import Scattering2D as Scattering


class Generator(nn.Module):
    def __init__(self, nb_input_channels, nb_hidden_channels, nb_output_channels=1):
        super(Generator, self).__init__()
        filter_size = 3
        padding = (filter_size - 1) // 2

        self.main = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(nb_input_channels, nb_hidden_channels, filter_size, bias=False),
            nn.BatchNorm2d(nb_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(nb_hidden_channels, nb_hidden_channels, filter_size, bias=False),
            nn.BatchNorm2d(nb_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(nb_hidden_channels, nb_output_channels, filter_size, bias=False),
            nn.BatchNorm2d(nb_output_channels, eps=0.001, momentum=0.9),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


def main():
    nb_epochs = 2

    transforms_to_apply = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Pixel values should be in [-1,1]
    ])

    dataset = datasets.MNIST('./temp/MNIST', train=True, download=True, transform=transforms_to_apply)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

    fixed_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    fixed_batch = next(iter(fixed_dataloader))
    fixed_batch = Variable(fixed_batch[0]).float().cuda()

    scat = Scattering(M=28, N=28, J=2)
    scat.cuda()

    scats_fixed_batch = scat(fixed_batch).squeeze(1)
    nb_input_channels = scats_fixed_batch.size(1)
    nb_hidden_channels = nb_input_channels

    generator = Generator(nb_input_channels, nb_hidden_channels)
    generator.cuda()
    generator.train()

    # Training of the network
    #########################

    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(generator.parameters())

    for idx_epoch in range(nb_epochs):
        print('Training epoch {}'.format(idx_epoch))
        for _, current_batch in enumerate(dataloader):
            generator.zero_grad()
            real_batch = Variable(current_batch[0]).float().cuda()
            scats_batch = scat(real_batch).squeeze(1)
            generated_batch = generator(scats_batch)
            loss = criterion(generated_batch, real_batch)
            loss.backward()
            optimizer.step()

    torch.save(generator.state_dict(), './temp/model.pth')

    generator.eval()

    # We create the batch containing the linear interpolation points in the scattering space
    ########################################################################################
    z0 = scats_fixed_batch.data.cpu().numpy()[[0]]
    z1 = scats_fixed_batch.data.cpu().numpy()[[1]]
    batch_z = np.copy(z0)
    nb_samples = 128
    interval = np.linspace(0, 1, nb_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))

    z = Variable(torch.from_numpy(batch_z)).float().cuda()
    path = generator(z).data.cpu().numpy().squeeze(1)
    path = (path + 1) / 2  # The pixels are now in [0, 1]

    # We store the nonlinear interpolation in the image space
    #########################################################
    os.mkdir('./temp/path/')
    for idx_image in range(nb_samples):
        filename = './temp/path/{}.png'.format(idx_image)
        Image.fromarray(np.uint8(path[idx_image] * 255.0)).save(filename)


if __name__ == '__main__':
    main()
