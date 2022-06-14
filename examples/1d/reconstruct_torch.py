"""
Reconstruct a synthetic signal from its scattering transform
============================================================
In this example we generate a harmonic signal of a few different frequencies,
analyze it with the 1D scattering transform, and reconstruct the scattering
transform back to the harmonic signal.
"""

###############################################################################
# Import the necessary packages
# -----------------------------

import numpy as np
import torch
from kymatio.torch import Scattering1D

from torch.autograd import backward
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

###############################################################################
# Write a function that can generate a harmonic signal
# ----------------------------------------------------
# Let's write a function that can generate some simple blip-type sounds with
# decaying harmonics. It will take four arguments: T, the length of the output
# vector; num_intervals, the number of different blips; gamma, the exponential
# decay factor of the harmonic; random_state, a random seed to generate
# random pitches and phase shifts.
# The function proceeds by splitting the time length T into intervals, chooses
# base frequencies and phases, generates sinusoidal sounds and harmonics,
# and then adds a windowed version to the output signal.
def generate_harmonic_signal(T, num_intervals=4, gamma=0.9, random_state=42):
    """
    Generates a harmonic signal, which is made of piecewise constant notes
    (of random fundamental frequency), with half overlap
    """
    rng = np.random.RandomState(random_state)
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

    return x

###############################################################################
# Let’s take a look at what such a signal could look like.

T = 2 ** 13
x = torch.from_numpy(generate_harmonic_signal(T))
plt.figure(figsize=(8, 2))
plt.plot(x.numpy())
plt.title("Original signal")

###############################################################################
# Let’s take a look at the signal spectrogram.

plt.figure(figsize=(8, 8))
plt.specgram(x.numpy(), Fs=1024)
plt.title("Spectrogram of original signal")

###############################################################################
## Doing the scattering transform.

J = 6
Q = 16

scattering = Scattering1D(J, T, Q).to(device)
x = x.to(device)

Sx = scattering(x)

learning_rate = 100
bold_driver_accelerator = 1.1
bold_driver_brake = 0.55
n_iterations = 200

###############################################################################
# Reconstruct the scattering transform back to original signal.

# Random guess to initialize.
torch.manual_seed(0)
y = torch.randn((T,), requires_grad=True, device=device)
Sy = scattering(y)

history = []
signal_update = torch.zeros_like(x, device=device)

# Iterate to recontsruct random guess to be close to target.
for k in range(n_iterations):
    # Backpropagation.
    err = torch.norm(Sx - Sy)

    if k % 10 == 0:
        print('Iteration %3d, loss %.2f' % (k, err.detach().cpu().numpy()))

    # Measure the new loss.
    history.append(err.detach().cpu())

    backward(err)

    delta_y = y.grad

    # Gradient descent
    with torch.no_grad():
        signal_update = - learning_rate * delta_y
        new_y = y + signal_update
    new_y.requires_grad = True

    # New forward propagation.
    Sy = scattering(new_y)

    if history[k] > history[k - 1]:
        learning_rate *= bold_driver_brake
    else:
        learning_rate *= bold_driver_accelerator
        y = new_y

plt.figure(figsize=(8, 2))
plt.plot(history)
plt.title("MSE error vs. iterations")

plt.figure(figsize=(8, 2))
plt.plot(y.detach().cpu().numpy())
plt.title("Reconstructed signal")

plt.figure(figsize=(8, 8))
plt.specgram(y.detach().cpu().numpy(), Fs=1024)
plt.title("Spectrogram of reconstructed signal")

plt.show()
