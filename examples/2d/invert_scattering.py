"""
Inverting scattering via mse
===========================
This script aims to quantify the information loss for natural images by
performing a reconstruction of an image from its scattering coefficients
via a L2-norm minimization.
"""

import numpy as np
import torch
import torch.nn.functional as F
from kymatio import Scattering2D
from torch import optim
import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"
img_name = "images/lena.jpg"

# Load src image
src_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
src_img = np.array(src_img).astype(np.float32)
y_true = np.array(src_img)
src_img = src_img / 255.
src_img = np.moveaxis(src_img, -1, 0)
print("img shape: ", src_img.shape)
channels, height, width = src_img.shape

for order in [1, 2]:
    for J in [3, 4, 5]:

        # Compute scattering coefficients
        scattering = Scattering2D(J=J, shape=(height, width), max_order=order)
        if device == "cuda":
            scattering = scattering.cuda()

        src_img_tensor = torch.from_numpy(src_img).to(device)
        scattering_coefficients = scattering(src_img_tensor)

        # Create random trainable input image
        input_tensor = torch.rand(
            src_img.shape, requires_grad=True, device=device)

        optimizer = optim.Adam([input_tensor], lr=10)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100,
                                                         verbose=True, threshold=1e-3, cooldown=50)

        # Training
        best_img = None
        best_loss = float("inf")
        for epoch in range(1, 1500):
            new_coefficients = scattering(input_tensor)
            loss = F.mse_loss(input=new_coefficients,
                              target=scattering_coefficients)
            scheduler.step(loss)
            print("Epoch {}, loss: {}".format(epoch, loss.item()), end='\r')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_loss = loss
                best_img = input_tensor.clone().detach()

        y_pred = best_img.cpu().detach().numpy()
        y_pred = np.moveaxis(y_pred, 0, -1)

        y_pred = np.clip(y_pred, 0., 1.)

        # PSNR
        print("")
        mse = np.mean((y_true - y_pred) ** 2)
        psnr = 20 * np.log10(1. / np.sqrt(mse))
        print("PSNR: {:.2f}dB for order {} and J={}".format(psnr, order, J))
