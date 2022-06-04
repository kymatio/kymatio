"""
Inverting scattering via mse
============================
This script aims to quantify the information loss for natural images by
performing a reconstruction of an image from its scattering coefficients via a
L2-norm minimization.
"""

###############################################################################
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from scipy.misc import face

from kymatio.torch import Scattering2D

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# Load test image
# ---------------
src_img = Image.fromarray(face())
src_img = src_img.resize((512, 384), Image.ANTIALIAS)
src_img = np.array(src_img).astype(np.float32)
src_img = src_img / 255.0
plt.imshow(src_img)
plt.title("Original image")

src_img = np.moveaxis(src_img, -1, 0)  # HWC to CHW
max_iter = 15 # number of steps for the GD
print("Image shape: ", src_img.shape)
channels, height, width = src_img.shape

###############################################################################
#  Main loop
# ----------
for order in [1]:
    for J in [2]:

        # Compute scattering coefficients
        scattering = Scattering2D(J=J, shape=(height, width), max_order=order)
        if device == "cuda":
            scattering = scattering.cuda()
            max_iter = 500
        src_img_tensor = torch.from_numpy(src_img).to(device).contiguous()
        scattering_coefficients = scattering(src_img_tensor)

        # Create trainable input image
        input_tensor = torch.rand(src_img.shape, requires_grad=True, device=device)

        # Optimizer hyperparams
        optimizer = optim.Adam([input_tensor], lr=1)

        # Training
        best_img = None
        best_loss = float("inf")
        for epoch in range(1, max_iter):
            new_coefficients = scattering(input_tensor)
            loss = F.mse_loss(input=new_coefficients, target=scattering_coefficients)
            print("Epoch {}, loss: {}".format(epoch, loss.item()), end="\r")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_loss = loss.detach().cpu().item()
                best_img = input_tensor.detach().cpu().numpy()

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
