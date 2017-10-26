"""
Author: angles
Date and Time: 25/10/17 - 14:24
"""

import os

import numpy as np
import scipy.misc as sm
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from scatwave.differentiable import scattering, cast, prepare_padding_size
from scatwave.filters_bank import filters_bank


def load(filename):
    return np.ascontiguousarray(Image.open(filename).convert('YCbCr'), dtype=np.uint8)


tr = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

im = load('0.png')
inputs = tr(im).unsqueeze(0).cuda()

M, N = 64, 64
J = 6

M_padded, N_padded = prepare_padding_size(M, N, J)

noise = torch.Tensor(inputs.size()).normal_(std=0.1).cuda()

filters = filters_bank(M_padded, N_padded, J)

Psi = filters['psi']
Phi = [filters['phi'][j] for j in range(J)]
Psi, Phi = cast(Psi, Phi, torch.cuda.FloatTensor)
x = Variable(noise, requires_grad=True)
Sy = scattering(Variable(inputs), Psi, Phi, J)

loss_vals = []


def f(a, b):
    S = scattering(a, Psi, Phi, J)
    return (torch.abs(S - b).mean(), S)  # +1e-4*(S**2).mean(),S


def closure():
    loss, Sx = f(x, Sy)
    print (Sx - Sy).abs().mean() / Sx.abs().mean()
    loss_vals.append(loss.data[0])
    loss.backward()
    return loss


optimizer = torch.optim.Adam([x], 0.1)

folder_to_save = os.path.expanduser('~/PycharmProjects/pyscatwave/')
from datetime import datetime

temp = 'J{}'.format(J)
folder_to_save = os.path.join(folder_to_save, temp + datetime.now().strftime("_%d%m%Y_%H%M%S"))


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


create_folder(folder_to_save)

for big_i in range(1000):
    for i in range(1):
        optimizer.zero_grad()
        optimizer.step(closure)
    I = (x.data.cpu()[0].permute(1, 2, 0).squeeze().numpy()).clip(0, 1)
    I = Image.fromarray(np.uint8(I * 255.0), 'YCbCr').convert('RGB')
    iteration = big_i * 1
    filename = os.path.join(folder_to_save, '{}.png'.format(iteration))
    I.save(filename)
