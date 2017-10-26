"""
Author: angles
Date and Time: 26/10/17 - 00:57
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from scatwave.differentiable import scattering, cast, prepare_padding_size
from scatwave.filters_bank import filters_bank

import scipy.misc as sm

path_scat = np.load('/home/angles/experiments/celebA_train_8192_test_2048_after_8192/paths/two_layers_4x4x1024_J6_model_2560_26102017_135541/scattering_coeffs.npy')
# path_scat = np.load('/home/angles/experiments/celebA_train_8192_test_2048_after_8192/temp_data/test_scat_J4.npy')
print(path_scat.shape)
path_scat = path_scat[1, :, :, :]
path_scat = np.transpose(path_scat, (2, 0, 1))
path_scat = path_scat.reshape((3, 417, 4, 4))
print(path_scat.shape)
nb_images = path_scat.shape[0]

# path_scat = np.load('scat.npy')
# print(path_scat.shape)
# path_scat = path_scat.transpose((2, 0, 1))
# path_scat = path_scat.reshape((3, 417, 4, 4))

M, N = 64, 64
J = 4

noise = torch.Tensor(1, 3, 64, 64).normal_(std=0.1).cuda()
x = Variable(noise, requires_grad=True)

Sy_tensor = torch.from_numpy(path_scat).type(torch.FloatTensor).unsqueeze(0).cuda()
Sy = Variable(Sy_tensor, requires_grad=False)
# print(Sy)

M_padded, N_padded = prepare_padding_size(M, N, J)

filters = filters_bank(M_padded, N_padded, J)

Psi = filters['psi']
Phi = [filters['phi'][j] for j in range(J)]

Psi, Phi = cast(Psi, Phi, torch.cuda.FloatTensor)
# print(noise)

loss_vals = []


def f(a, b):
    S = scattering(a, Psi, Phi, J)
    # print(S.size())
    return torch.abs(S - b).mean(), S  # +1e-4*(S**2).mean(),S


def closure():
    loss, Sx = f(x, Sy)
    print (Sx - Sy).abs().mean() / Sx.abs().mean()
    loss_vals.append(loss.data[0])
    loss.backward()
    return loss


optimizer = torch.optim.Adam([x], 0.1)

folder_to_save = os.path.expanduser('~/PycharmProjects/pyscatwave/')
from datetime import datetime

temp = 'J{}_{}_img{}'.format(J, 'test_path', 'Random')
folder_to_save = os.path.join(folder_to_save, temp + datetime.now().strftime("_%d%m%Y_%H%M%S"))


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


create_folder(folder_to_save)

for big_i in range(1000):
    for i in tqdm(range(10)):
        optimizer.zero_grad()
        optimizer.step(closure)
    I = (x.data.cpu()[0].permute(1, 2, 0).squeeze().numpy()).clip(0, 1)
    print(I.shape)
    # L = (inputs.cpu()[0].permute(1, 2, 0).squeeze().numpy()).clip(0, 1)
    # plt.imshow(I)
    # plt.show()
    I = Image.fromarray(np.uint8(I * 255.0), 'RGB')
    # # I = Image.fromarray(np.uint8(I * 255.0), 'RGB')
    # # L = Image.fromarray(np.uint8(L * 255.0), 'YCbCr').convert('RGB')
    iteration = big_i * 10
    filename = os.path.join(folder_to_save, '{}.png'.format(iteration))
    I.save(filename)
    # # plt.imshow(I)
    # # # plt.imshow(L)
    # # plt.show()
