"""
Author: angles
Date and Time: 26/10/17 - 03:53
"""

import os
from glob import glob

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
    return np.ascontiguousarray(Image.open(filename), dtype=np.uint8)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def compute_scat_from_dataset(folder_images, images_extension, folder_temp_data, test_or_train, J):
    assert test_or_train in ['train', 'test']

    files = glob(os.path.join(folder_images, '*.{0}'.format(images_extension)))
    nb_images = len(files)
    temp = sm.imread(files[0])
    assert temp.shape[0] == temp.shape[1], "The images should be square."

    M, N = 64, 64
    M_padded, N_padded = prepare_padding_size(M, N, J)

    tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    filters = filters_bank(M_padded, N_padded, J)

    Psi = filters['psi']
    Phi = [filters['phi'][j] for j in range(J)]

    Psi, Phi = cast(Psi, Phi, torch.cuda.FloatTensor)

    batch_size = 512
    nb_batches = nb_images // batch_size
    size = 2 ** (6 - J)
    nb_channels = 1 + J * 8 + J * (J - 1) * 32  # L=8
    total = np.zeros((nb_images, size, size, 3 * nb_channels))  # color
    for idx_batch in tqdm(range(nb_batches)):
        inputs = torch.Tensor(batch_size, 3, 64, 64).cuda()
        for idx_image in range(batch_size):
            idx = idx_batch * batch_size + idx_image
            filename = os.path.join(folder_images, '{0}.{1}'.format(idx, images_extension))
            temp = load(filename)
            inputs[idx_image] = tr(temp)
        S = scattering(Variable(inputs), Psi, Phi, J)
        S_npy = S.data.cpu().numpy()
        # print(S_npy.shape)
        S_npy = S_npy.reshape((batch_size, -1, size, size))
        S_npy = S_npy.transpose((0, 2, 3, 1))
        total[idx_batch * batch_size:(idx_batch + 1) * batch_size, :, :, :] = S_npy

    if np.isnan(total).any():
        print(' [!] There is at least a NaN in the tensor.')
        return
    filename = os.path.join(folder_temp_data, '{0}_scat_J{1}_.npy'.format(test_or_train, J))
    np.save(filename, total)


if __name__ == '__main__':
    dataset = 'convex_shapes'
    images_extension = 'png'
    train_attribute = 'five_points_8192_rgb'
    test_attribute = 'five_points_2048_rgb'

    global_folder = os.path.expanduser('~/experiments/')
    partial_folder_dataset = dataset + '_train_' + train_attribute + '_test_' + test_attribute
    global_folder = os.path.join(global_folder, partial_folder_dataset)
    folder_temp_data = os.path.join(global_folder, 'temp_data')

    create_folder(folder_temp_data)
    folder_dataset = os.path.expanduser('~/datasets/{0}'.format(dataset))
    folder_train_images = os.path.join(folder_dataset, train_attribute)
    folder_test_images = os.path.join(folder_dataset, test_attribute)

    for J in [4, 5]:
        compute_scat_from_dataset(folder_train_images, images_extension, folder_temp_data, 'train', J)
        compute_scat_from_dataset(folder_test_images, images_extension, folder_temp_data, 'test', J)
