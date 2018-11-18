"""
Plot of the 1D Morlet filters used for the Scattering Transform
===============================================================
See :meth:`scattering.scattering1d.filter_bank` for more informations about the wavelets used.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from kymatio.scattering1d.filter_bank import scattering_filter_factory


#######################################
# Initial parameters of the filter bank
# -------------------------------------

T = 2**13
J = 6
Q = 8

phi_f, psi1_f, psi2_f, _ = scattering_filter_factory(math.log2(T), J, Q)

###############################################################################
# Plot first-order (Q = 8) and second-order (Q = 1) wavelets along with their
# lowpass filters
# ---------------------------------------------------------------------------

fig, axs = plt.subplots(2, 1)

psis_f = (psi1_f, psi2_f)
Qs = (Q, 1)

for k in range(2):
    axs[k].plot(np.arange(T)/T, phi_f[0], 'b')
    for psi_f in psis_f[k]:
        axs[k].plot(np.arange(T)/T, psi_f[0], 'b')
    axs[k].set_xlim(0, 0.5)
    axs[k].set_ylim(0, 1.2)
    axs[k].set_xlabel('\omega')
    axs[k].set_ylabel('\hat\psi_j(\omega)')
    axs[k].set_title('Q = {}'.format(Qs[k]))

fig.suptitle(("Fourier transforms of wavelets for all scales j with the "
    "corresponding lowpass filter."))

plt.show()
