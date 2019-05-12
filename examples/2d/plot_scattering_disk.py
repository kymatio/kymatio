"""
Scattering disk display
=======================

Description:
It reproduces concentric circles that encode Scattering coefficient's energy as described in
"Invariant Scattering Convolution Networks" by Bruna and Mallat.

Author:
https://github.com/Jonas1312

Edited by:
Edouard Oyallon
"""

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from kymatio import Scattering2D
from PIL import Image



device = "cuda" if torch.cuda.is_available() else "cpu"
img_name = "images/3.JPG"

####################################################################
# Routines to compute the Scattering Transform
#-------------------------------------------------------------------
# Read image
src_img = Image.open(img_name).convert('L')
src_img = np.array(src_img)
print("img shape: ", src_img.shape)

####################################################################
# Create scattering object
L = 12
J = 3
scattering = Scattering2D(J=J, shape=src_img.shape, L=L, max_order=1)
if device == "cuda":
    scattering = scattering.cuda()

####################################################################
# Compute scattering coefficients
src_img = src_img.astype(np.float32) / 255.
src_img_tensor = torch.from_numpy(src_img).to(device)
scattering_coefficients = scattering(src_img_tensor)
print("coeffs shape: ", scattering_coefficients.size())

####################################################################
# Routines to display the Scattering disk
#-------------------------------------------------------------------
#
scattering_coefficients = scattering_coefficients.cpu().numpy()
# Invert colors
scattering_coefficients = -scattering_coefficients

####################################################################
# Skip the low pass filter
scattering_coefficients = scattering_coefficients[1:, :, :]
norm = mpl.colors.Normalize(scattering_coefficients.min(), scattering_coefficients.max(), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap="gray")
nb_coeffs, window_rows, window_columns = scattering_coefficients.shape
fig, axs = plt.figure(figsize=(9, 9))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)

####################################################################
# White circle for consistency with the original display
offset = 0.1
for row in range(window_rows):
    for column in range(window_columns):
        ax = plt.subplot(window_rows, window_columns, 1 + column + row * window_rows, projection='polar')
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_yticklabels([])  # turn off radial tick labels (yticks)
        ax.set_xticklabels([])  # turn off degrees
        # ax.set_theta_zero_location('N')  # 0° to North
        coefficients = scattering_coefficients[:, row, column]
        for j in range(J):
            for l in range(L):
                coeff = coefficients[l + (J - 1 - j) * L]
                color = mpl.colors.to_hex(mapper.to_rgba(coeff))
                ax.bar(x=l * 2 * np.pi / L,
                       height=(1 - offset) / J,
                       width=2 * np.pi / L,
                       bottom=offset + (j / J) * (1 - offset),
                       color=color)
fig.show()