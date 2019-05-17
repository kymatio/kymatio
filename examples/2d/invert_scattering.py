"""
Inverting scattering via mse
===========================
This script aims to quantify the information loss for natural images by
performing a reconstruction of an image from its scattering coefficients
via a L2-norm minimization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim

from kymatio import Scattering2D

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load src image
img_name = "./images/lena_64.png"
src_img = Image.open(img_name).convert("RGB")
src_img = np.array(src_img).astype(np.float32)
src_img = src_img / 255.0
y_true = np.array(src_img)  # copy of src_img
plt.imshow(y_true)
plt.title("Original image")

src_img = np.moveaxis(src_img, -1, 0)  # HWC to CHW
print("img shape: ", src_img.shape)
channels, height, width = src_img.shape

for order in [1]:
    for J in [2, 5]:

        # Compute scattering coefficients
        scattering = Scattering2D(J=J, shape=(height, width), max_order=order)
        if device == "cuda":
            scattering = scattering.cuda()
        src_img_tensor = torch.from_numpy(src_img).to(device)
        scattering_coefficients = scattering(src_img_tensor)

        # Create trainable input image
        input_tensor = torch.rand(src_img.shape, requires_grad=True, device=device)

        # Optimizer hyperparams
        optimizer = optim.Adam([input_tensor], lr=1)

        # Training
        best_img = None
        best_loss = float("inf")
        for epoch in range(1, 300):
            new_coefficients = scattering(input_tensor)
            loss = F.mse_loss(input=new_coefficients, target=scattering_coefficients)
            print("Epoch {}, loss: {}".format(epoch, loss.item()), end="\r")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_loss = loss
                best_img = input_tensor.clone().detach()

        y_pred = best_img.cpu().detach().numpy()
        y_pred = np.moveaxis(y_pred, 0, -1)  # CHW to HWC

        y_pred = np.clip(y_pred, 0.0, 1.0)

        # PSNR
        print("")
        mse = np.mean((y_true - y_pred) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        print("PSNR: {:.2f}dB for order {} and J={}".format(psnr, order, J))

        # Plot
        plt.figure()
        plt.imshow(y_pred)
        plt.title("PSNR: {:.2f}dB (order {}, J={})".format(psnr, order, J))


plt.show()
