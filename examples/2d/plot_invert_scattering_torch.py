"""
Inverting scattering via MSE
============================
This script aims to quantify the information loss for natural images by
performing a reconstruction of an image from its scattering coefficients via a
L2-norm minimization. Note that this is only possible because we use a small
averaging scale (:math:`2 ^ J = 4`). For more information, see Section 3.4 in
*Scattering Networks for Hybrid Representation Learning*.
"""

###############################################################################
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import face

import torch
import torch.nn.functional as F
from torch import optim

from kymatio.torch import Scattering2D

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# Load test image
# ---------------
src_img = Image.fromarray(face())
src_img = src_img.resize((512, 384), Image.Resampling.LANCZOS)
src_img = np.array(src_img).astype(np.float32)
src_img = src_img / 255.0

plt.imshow(src_img)
plt.title("Original image")

src_img = np.moveaxis(src_img, -1, 0)  # HWC to CHW
print("Image shape: ", src_img.shape)
channels, height, width = src_img.shape

###############################################################################
#  Main loop
# ----------
order = 1
J = 2
max_iter = 50

# Compute scattering coefficients
scattering = Scattering2D(J=J, shape=(height, width), max_order=order)
scattering.to(device)
src_img_tensor = torch.from_numpy(src_img).to(device).contiguous()
scattering_coefficients = scattering(src_img_tensor)

# Initialize image
img = torch.rand(src_img.shape, requires_grad=True, device=device)

# Set up optimizer
optimizer = optim.Adam([img], lr=1)

# Training
best_img = None
best_loss = float("inf")

for epoch in range(max_iter):
    new_coefficients = scattering(img)
    loss = F.mse_loss(input=new_coefficients, target=scattering_coefficients)
    print("Epoch {}, loss: {}".format(epoch + 1, loss.item()), end="\r")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss < best_loss:
        best_loss = loss.detach().cpu().item()
        best_img = img.detach().cpu().numpy()

best_img = np.clip(best_img, 0.0, 1.0)

# PSNR
mse = np.mean((src_img - best_img) ** 2)
psnr = 20 * np.log10(1.0 / np.sqrt(mse))
print("\nPSNR: {:.2f}dB for order {} and J={}".format(psnr, order, J))

# Plot
plt.figure()
plt.imshow(np.moveaxis(best_img, 0, -1))
plt.title("PSNR: {:.2f}dB (order {}, J={})".format(psnr, order, J))

plt.show()
