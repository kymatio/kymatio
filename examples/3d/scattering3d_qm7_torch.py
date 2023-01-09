"""
3D scattering quantum chemistry regression
==========================================

Description:
This example trains a classifier combined with a scattering transform to
regress molecular atomization energies on the QM7 dataset. Here, we use full
charges, valence charges and core charges. A linear regression is deployed.

Remarks:
The linear regression of the QM7 energies with the given values gives MAE
2.75, RMSE 4.18 (kcal.mol-1)

Reference:
https://arxiv.org/abs/1805.00571
"""

###############################################################################
# Preliminaries
# -------------
#
# First, we import NumPy, PyTorch, and some utility modules.

import numpy as np
import torch
import time
import os

###############################################################################
# We will use scikit-learn to construct a linear model, so we import the
# necessary modules. In addition, we need to compute distance matrices when
# normalizing our input features, so we import `pdist` from `scipy.spatial`.

from sklearn import (linear_model, model_selection, preprocessing,
                     pipeline)
from scipy.spatial.distance import pdist

######################################################################
# We then import the necessary functionality from Kymatio. First, we need the
# PyTorch frontend of the 3D solid harmonic cattering transform.

from kymatio.torch import HarmonicScattering3D

###############################################################################
# The 3D transform doesn't compute the zeroth-order coefficients, so we need
# to import `compute_integrals` to do this manually.

from kymatio.scattering3d.backend.torch_backend \
    import TorchBackend3D

###############################################################################
# To generate the input 3D maps, we need to calculate sums of Gaussians, so we
# import the function `generate_weighted_sum_of_gaussians`.

from kymatio.scattering3d.utils \
    import generate_weighted_sum_of_gaussians

###############################################################################
# Finally, we import the utility functions that let us access the QM7 dataset
# and the cache directories to store our results.

from ...datasets import fetch_qm7
from kymatio.caching import get_cache_dir

###############################################################################
# Data preparation
# ----------------
#
# Fetch the QM7 database and extract the atomic positions and nuclear charges
# of each molecule. This dataset contains 7165 organic molecules with up to
# seven non-hydrogen atoms, whose energies were computed using density
# functional theory.

qm7 = fetch_qm7(align=True)
pos = qm7['positions']
full_charges = qm7['charges']

n_molecules = pos.shape[0]

###############################################################################
# From the nuclear charges, we compute the number of valence electrons, which
# we store as the valence charge of that atom.

mask = full_charges <= 2
valence_charges = full_charges * mask

mask = np.logical_and(full_charges > 2, full_charges <= 10)
valence_charges += (full_charges - 2) * mask

mask = np.logical_and(full_charges > 10, full_charges <= 18)
valence_charges += (full_charges - 10) * mask

###############################################################################
# We then normalize the positions of the atoms. Specifically, the positions
# are rescaled such that two Gaussians of width `sigma` placed at those
# positions overlap with amplitude less than `overlapping_precision`.

overlapping_precision = 1e-1
sigma = 2.0
min_dist = np.inf

for i in range(n_molecules):
    n_atoms = np.sum(full_charges[i] != 0)
    pos_i = pos[i, :n_atoms, :]
    min_dist = min(min_dist, pdist(pos_i).min())

delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
pos = pos * delta / min_dist

###############################################################################
# Scattering Transform
# --------------------------------
# Given the rescaled positions and charges, we are now ready to compute the
# density maps by placing Gaussians at the different positions weighted by the
# appropriate charge. These are fed into the 3D solid harmonic scattering
# transform to obtain features that are used to regress the energies. In
# order to do this, we must first define a grid.

M, N, O = 192, 128, 96

grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O]
grid = np.fft.ifftshift(grid)

###############################################################################
# We then define the scattering transform using the `HarmonicScattering3D`
# class.

J = 2
L = 3
integral_powers = [0.5, 1.0, 2.0, 3.0]

scattering = HarmonicScattering3D(J=J, shape=(M, N, O),
                                  L=L, sigma_0=sigma,
                                  integral_powers=integral_powers)

###############################################################################
# We then check whether a GPU is available, in which case we transfer our
# scattering object there.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
scattering.to(device)

###############################################################################
# The maps computed for each molecule are quite large, so the computation has
# to be done by batches. Here we select a small batch size to ensure that we
# have enough memory when running on the GPU. Dividing the number of molecules
# by the batch size then gives us the number of batches.

batch_size = 8
n_batches = int(np.ceil(n_molecules / batch_size))

###############################################################################
# We are now ready to compute the scattering transforms. In the following
# loop, each batch of molecules is transformed into three maps using Gaussians
# centered at the atomic positions, one for the nuclear charges, one for the
# valence charges, and one with their difference (called the “core” charges).
# For each map, we compute its scattering transform up to order two and store
# the results.

order_0, orders_1_and_2 = [], []
print('Computing solid harmonic scattering coefficients of '
      '{} molecules from the QM7 database on {}'.format(
        n_molecules,   "GPU" if use_cuda else "CPU"))
print('sigma: {}, L: {}, J: {}, integral powers: {}'.format(
        sigma, L, J, integral_powers))

this_time = None
last_time = None
for i in range(n_batches):
    this_time = time.time()
    if last_time is not None:
        dt = this_time - last_time
        print("Iteration {} ETA: [{:02}:{:02}:{:02}]".format(
                    i + 1, int(((n_batches - i - 1) * dt) // 3600),
                    int((((n_batches - i - 1) * dt) // 60) % 60),
                    int(((n_batches - i - 1) * dt) % 60)))
    else:
        print("Iteration {} ETA: {}".format(i + 1, '-'))
    last_time = this_time
    time.sleep(1)

    # Extract the current batch.
    start = i * batch_size
    end = min(start + batch_size, n_molecules)

    pos_batch = pos[start:end]
    full_batch = full_charges[start:end]
    val_batch = valence_charges[start:end]

    # Calculate the density map for the nuclear charges and transfer
    # to PyTorch.
    full_density_batch = generate_weighted_sum_of_gaussians(grid,
            pos_batch, full_batch, sigma)
    full_density_batch = torch.from_numpy(full_density_batch)
    full_density_batch = full_density_batch.to(device).float()

    # Compute zeroth-order, first-order, and second-order scattering
    # coefficients of the nuclear charges.
    full_order_0 = TorchBackend3D.compute_integrals(full_density_batch,
                                     integral_powers)
    full_scattering = scattering(full_density_batch)

    # Compute the map for valence charges.
    val_density_batch = generate_weighted_sum_of_gaussians(grid,
            pos_batch, val_batch, sigma)
    val_density_batch = torch.from_numpy(val_density_batch)
    val_density_batch = val_density_batch.to(device).float()

    # Compute scattering coefficients for the valence charges.
    val_order_0 = TorchBackend3D.compute_integrals(val_density_batch,
                                    integral_powers)
    val_scattering = scattering(val_density_batch)

    # Take the difference between nuclear and valence charges, then
    # compute the corresponding scattering coefficients.
    core_density_batch = full_density_batch - val_density_batch

    core_order_0 = TorchBackend3D.compute_integrals(core_density_batch,
                                     integral_powers)
    core_scattering = scattering(core_density_batch)

    # Stack the nuclear, valence, and core coefficients into arrays
    # and append them to the output.
    batch_order_0 = torch.stack(
        (full_order_0, val_order_0, core_order_0), dim=-1)
    batch_orders_1_and_2 = torch.stack(
        (full_scattering, val_scattering, core_scattering), dim=-1)

    order_0.append(batch_order_0)
    orders_1_and_2.append(batch_orders_1_and_2)

###############################################################################
# Concatenate the batch outputs and transfer to NumPy.

order_0 = torch.cat(order_0, dim=0)
orders_1_and_2 = torch.cat(orders_1_and_2, dim=0)

order_0 = order_0.cpu().numpy()
orders_1_and_2 = orders_1_and_2.cpu().numpy()

###############################################################################
# Regression
# ----------
#
# To use the scattering coefficients as features in a scikit-learn pipeline,
# these must be of shape `(n_samples, n_features)`, so we reshape our arrays
# accordingly.

order_0 = order_0.reshape((n_molecules, -1))
orders_1_and_2 = orders_1_and_2.reshape((n_molecules, -1))

###############################################################################
# Since the above calculation is quite lengthy, we save the results to a cache
# for future use.

basename = 'qm7_L_{}_J_{}_sigma_{}_MNO_{}_powers_{}.npy'.format(
        L, J, sigma, (M, N, O), integral_powers)

cache_dir = get_cache_dir("qm7/experiments")

filename = os.path.join(cache_dir, 'order_0_' + basename)
np.save(filename, order_0)

filename = os.path.join(cache_dir, 'orders_1_and_2' + basename)
np.save(filename, orders_1_and_2)

###############################################################################
# We now concatenate the zeroth-order coefficients with the rest since we want
# to use all of them as features.

scattering_coef = np.concatenate([order_0, orders_1_and_2], axis=1)

###############################################################################
# Fetch the target energies from the QM7 dataset.

qm7 = fetch_qm7()
target = qm7['energies']

###############################################################################
# We evaluate the performance of the regression using five-fold
# cross-validation. To do so, we first shuffle the molecules, then we store
# the resulting indices in `cross_val_folds`.

n_folds = 5

P = np.random.permutation(n_molecules).reshape((n_folds, -1))

cross_val_folds = []

for i_fold in range(n_folds):
    fold = (np.concatenate(P[np.arange(n_folds) != i_fold], axis=0),
            P[i_fold])
    cross_val_folds.append(fold)

###############################################################################
# Given these folds, we compute the regression error for various settings of
# the `alpha` parameter, which controls the amount of regularization applied
# to the regression problem (here in the form of a simple ridge regression, or
# Tikhonov, regularization). The mean absolute error (MAE) and root mean
# square error (RMSE) is output for each value of `alpha`.

alphas = 10.0 ** (-np.arange(1, 10))
for i, alpha in enumerate(alphas):
    scaler = preprocessing.StandardScaler()
    ridge = linear_model.Ridge(alpha=alpha)

    regressor = pipeline.make_pipeline(scaler, ridge)

    target_prediction = model_selection.cross_val_predict(regressor,
            X=scattering_coef, y=target, cv=cross_val_folds)

    MAE = np.mean(np.abs(target_prediction - target))
    RMSE = np.sqrt(np.mean((target_prediction - target) ** 2))

    print('Ridge regression, alpha: {}, MAE: {}, RMSE: {}'.format(
        alpha, MAE, RMSE))
