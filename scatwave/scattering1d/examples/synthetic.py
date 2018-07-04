import torch
from torch.autograd import Variable
from scatwave import Scattering1D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state


def generate_harmonic_signal(T, num_intervals=4, gamma=0.9, random_state=42):
    """
    Generates a harmonic signal, which is made of piecewise constant notes
    (of random fundamental frequency), with half overlap
    """
    rng = check_random_state(random_state)
    num_notes = 2 * (num_intervals - 1) + 1
    support = T // num_intervals
    half_support = support // 2

    base_freq = 0.1 * rng.rand(num_notes) + 0.05
    phase = 2 * np.pi * rng.rand(num_notes)
    window = np.hanning(support)
    x = np.zeros(T, dtype='float32')
    t = np.arange(0, support)
    u = 2 * np.pi * t
    for i in range(num_notes):
        ind_start = i * half_support
        note = np.zeros(support)
        for k in range(1):
            note += (np.power(gamma, k) *
                     np.cos(u * (k + 1) * base_freq[i] + phase[i]))
        x[ind_start:ind_start + support] += note * window
    # put x in a Variable
    x = Variable(torch.from_numpy(x[np.newaxis, np.newaxis]))
    return x


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
    # Scattering definition
    T = 2**13
    J = 6
    Q = 16
    scattering = Scattering1D(T, J, Q)

    # get the metadata on the coordinates of the scattering
    coords = Scattering1D.compute_meta_scattering(J, Q, order2=True)
    order0 = torch.LongTensor([0])
    order1 = torch.LongTensor(
        sorted([cc for cc in coords.keys() if coords[cc]['order'] == '1']))
    order2 = torch.LongTensor(
        sorted([cc for cc in coords.keys() if coords[cc]['order'] == '2']))

    # harmonic signal
    x = generate_harmonic_signal(T)
    s = scattering.forward(x).data[0]
    show_signal(x, s, order0, order1, order2)
