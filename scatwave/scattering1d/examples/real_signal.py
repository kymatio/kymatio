import torch
from torch.autograd import Variable
from scatwave import Scattering1D
from scatwave import fetch_fsdd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os


def loadfile(path_file):
    sr, x = wavfile.read(path_file)
    x = np.asarray(x, dtype='float')
    # make it mono
    if x.ndim > 1:
        smallest_axis = np.argmin(x.shape)
        x = x.mean(axis=smallest_axis)
    x = np.asarray(x, dtype='float')
    x /= np.max(np.abs(x))
    return sr, x


def show_signal(x, s, order0, order1, order2):
    fig, axarr = plt.subplots(4, 1, figsize=(8, 16))
    axarr[0].plot(x.data[0, 0])
    axarr[0].set_title('Original signal')
    axarr[1].plot(s[order0][0])
    axarr[1].set_title('Scattering Order 0')
    axarr[2].imshow(s[order1], aspect='auto')
    axarr[2].set_title('Scattering Order 1')
    axarr[3].imshow(s[order2], aspect='auto')
    axarr[3].set_title('Scattering Order 2')
    plt.show()


if __name__ == '__main__':
    # fetch the dataset and get the signal
    info_dataset = fetch_fsdd(base_dir='fsdd', verbose=True)
    filepath = os.path.join(info_dataset['path_dataset'],
                            sorted(info_dataset['files'])[0])

    # Load the signal
    sr, x = loadfile(filepath)
    x_th = Variable(torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0))

    # Prepare the scattering
    T = x_th.shape[-1]
    J = 6
    Q = 16
    scattering = Scattering1D(T, J, Q)

    # Get the metadata
    coords = Scattering1D.compute_meta_scattering(J, Q, order2=True)
    order0 = torch.LongTensor([0])
    order1 = torch.LongTensor(
        sorted([cc for cc in coords.keys() if coords[cc]['order'] == '1']))
    order2 = torch.LongTensor(
        sorted([cc for cc in coords.keys() if coords[cc]['order'] == '2']))

    # Compute the scattering
    s = scattering.forward(x_th).data.numpy()[0]

    # show it
    show_signal(x_th, s, order0, order1, order2)
