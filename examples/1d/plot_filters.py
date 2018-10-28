"""
Plot of the 1D Morlet filters used for the Scattering Transform
===============================
See :ref:`filter_bank` for more informations about the wavelets used.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scattering.scattering1d.filter_bank import scattering_filter_factory


###############################################################################
# Initial parameters of the filter bank
# -------------------------------------

T = 2**13
J = 6
Q = 8

phi_fft, psi1_fft, psi2_fft, _ = scattering_filter_factory(math.log2(T), J, Q)

###############################################################################
# Plot first-order (Q = 8) and second-order (Q = 1) wavelets along with their
# lowpass filters
# -------------------------------------

fig, axs = plt.subplots(2, 1)

psis_fft = (psi1_fft, psi2_fft)
Qs = (Q, 1)

for k in range(2):
    axs[k].plot(np.arange(T)/T, phi_fft[0], 'b')
    for key in psis_fft[k].keys():
        axs[k].plot(np.arange(T)/T, psis_fft[k][key][0], 'b')
    axs[k].set_xlim(0, 0.5)
    axs[k].set_ylim(0, 1.2)
    axs[k].set_xlabel('\omega')
    axs[k].set_ylabel('\hat\psi_j(\omega)')
    axs[k].set_title('Q = {}'.format(Qs[k]))

fig.suptitle(("Fourier transforms of wavelets for all scales j with the "
    "corresponding lowpass filter."))

plt.show()
