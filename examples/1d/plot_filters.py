"""
Plot of the 1D Morlet filters used for the Scattering Transform
===============================
See :ref:`filter_bank` for more informations about the used wavelets.
"""

import numpy as np
import matplotlib.pyplot as plt
from scattering.scattering2d.filter_bank import filter_bank
import scipy.fftpack as fft


###############################################################################
# Initial parameters of the filter bank
# -------------------------------------

J_support = 3
J_scattering = 8
Q = 7

phi_fft, psi1_fft, psi2_fft, _ = scattering_filter_factory(J_support, J_scattering, Q)




fig, axs = plt.subplots(J+1, L, sharex=True, sharey=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

###############################################################################
# Bandpass filters $\psi_1$
# ----------------
# First, we display each wavelets according to each scale and orientation.
i=0
for k in len(psi1_fft):
    #f_r = filter[0][...,0].numpy()
    #f_i = filter[0][..., 1].numpy()
    #f = f_r + 1j*f_i
    #filter_c = fft.fft(f)
    filter_c = psi1_fft[k]
    filter_c = np.fft.fftshift(filter_c)
    axs[i // L, i % L].plot(filter_c)
    axs[i // L, i % L].set_title("$j = {}$ \n $\\theta={}".format(i // L, i % L))
    i = i+1


# Add blanks for pretty display
for z in range(L):
    axs[i // L, i % L].axis('off')
    i = i+1

###############################################################################
# Lowpass filter
# ----------------
# We finally display the Gaussian filter.
f_r = filters_set['phi'][0][...,0].numpy()
f_i = filters_set['phi'][0][..., 1].numpy()
f = f_r + 1j*f_i
filter_c = fft.fft2(f)
filter_c = np.fft.fftshift(filter_c)
axs[J, L // 2].imshow(colorize(filter_c))

# Final caption.
fig.suptitle("Wavelets for each scales $j$ and angles $\\theta$ used, with the corresponding low-pass filter."
             "\n The contrast corresponds to the amplitude and the color to the phase.", fontsize=13)


plt.show()