"""
Plot the 2D wavelet filters
===========================
See :meth:`kymatio.scattering2d.filter_bank` for more information about the wavelets.
"""

import numpy as np
import matplotlib.pyplot as plt

from kymatio.scattering2d.filter_bank import filter_bank


###############################################################################
# Initial parameters of the filter bank
# -------------------------------------
# Images are of shape `(M, M) = (32, 32)`, maximum wavelet scale is
# `2 ** J = 2 ** 3 = 8` and number of orientations is `L = 8`.

M = 32
J = 3
L = 8

# Calculate the filters (in the frequency domain).
filters_set = filter_bank(M, M, J, L=L)

###############################################################################
# Imshow complex images
# ---------------------
# Since the wavelets are complex, we need to colorize them to show them
# properly (see
# https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array)

from colorsys import hls_to_rgb

def colorize(im):
    # Normalize input to have largest magnitude one.
    im = im / np.max(np.abs(im))

    # Use complex phase to determine hue (between zero and one).
    A = (np.angle(im) + np.pi) / (2 * np.pi)
    A = (A + 0.5) % 1.0

    # Use magnitude to determine lightness.
    B = 1.0 / (1.0 + np.abs(im) ** 0.3)

    # Put everything together and convert to RGB.
    A = A.flatten()
    B = B.flatten()

    im_c = np.array([hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)])
    im_c = im_c.reshape(im.shape + (3,))

    return im_c

###############################################################################
# Bandpass filters
# ----------------
# Display each wavelet according to its scale and orientation. Color (hue)
# denotes complex phase, while lightness denotes inverse magnitude.

fig, axs = plt.subplots(J, L, sharex=True, sharey=True)

for k, psi_f in enumerate(filters_set['psi']):
    psi_f = psi_f["levels"][0]
    psi = np.fft.ifft2(psi_f)
    psi = np.fft.fftshift(psi)

    j = k // L
    theta = k % L

    axs[j, theta].imshow(colorize(psi))
    axs[j, theta].axis('off')
    axs[j, theta].set_title(f"$j = {j}$ \n $\\theta={theta}$")

fig.suptitle("Wavelets for each scale $j$ and angle $\\theta$")
plt.show()

###############################################################################
# Lowpass filter
# --------------
# Now display the low-pass filter.

phi_f = filters_set['phi']["levels"][0]

phi = np.fft.ifft2(phi_f)
phi = np.fft.fftshift(phi)
plt.suptitle("The corresponding low-pass filter (scaling function)")

plt.imshow(np.abs(phi), cmap="gray_r")
plt.show()
