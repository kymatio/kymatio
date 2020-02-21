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

import numpy as np
import time
import torch
import os

from sklearn import linear_model, model_selection, preprocessing, pipeline
from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians
from kymatio.scattering3d.backend.torch_backend import compute_integrals
from kymatio.datasets import fetch_qm7
from kymatio.caching import get_cache_dir
from scipy.spatial.distance import pdist



M, N, O, J, L = 192, 128, 96, 2, 3
integral_powers = [0.5,  1., 2., 3.]
sigma = 2.

batch_size = 8

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O]
grid = grid.astype('float32')
grid = np.fft.ifftshift(grid, axes=(1, 2, 3))

qm7 = fetch_qm7(align=True)
pos = qm7['positions']
full_charges = qm7['charges'].astype('float32')

mask = full_charges <= 2
valence_charges = full_charges * mask

mask = np.logical_and(full_charges > 2, full_charges <= 10)
valence_charges += (full_charges - 2) * mask

mask = np.logical_and(full_charges > 10, full_charges <= 18)
valence_charges += (full_charges - 10) * mask

# normalize positions
overlapping_precision = 1e-1
min_dist = np.inf
for i in range(pos.shape[0]):
    n_atoms = np.sum(full_charges[i] != 0)
    pos_i = pos[i, :n_atoms, :]
    min_dist = min(min_dist, pdist(pos_i).min())
delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
pos = pos * delta / min_dist

n_molecules = pos.shape[0]
n_batches = np.ceil(n_molecules / batch_size).astype(int)

scattering = HarmonicScattering3D(J=J, shape=(M, N, O), L=L, sigma_0=sigma).to(device)

order_0, orders_1_and_2 = [], []
print('Computing solid harmonic scattering coefficients of {} molecules '
    'of QM7 database on {}'.format(pos.shape[0], 'GPU' if device == 'cuda' else 'CPU'))
print('sigma: {}, L: {}, J: {}, integral powers: {}'.format(sigma, L, J, integral_powers))

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

    start, end = i * batch_size, min((i + 1) * batch_size, n_molecules)

    pos_batch = pos[start:end]
    full_batch = full_charges[start:end]
    val_batch = valence_charges[start:end]

    full_density_batch = generate_weighted_sum_of_gaussians(grid,
            pos_batch, full_batch, sigma)
    full_density_batch = torch.from_numpy(full_density_batch)
    full_density_batch = full_density_batch.to(device).float()

    full_order_0 = compute_integrals(full_density_batch, integral_powers)
    scattering.max_order = 2
    scattering.method = 'integral'
    scattering.integral_powers = integral_powers
    full_scattering = scattering(full_density_batch)

    val_density_batch = generate_weighted_sum_of_gaussians(grid,
            pos_batch, val_batch, sigma)
    val_density_batch = torch.from_numpy(val_density_batch)
    val_density_batch = val_density_batch.to(device).float()

    val_order_0 = compute_integrals(val_density_batch, integral_powers)
    val_scattering = scattering(val_density_batch)

    core_density_batch = full_density_batch - val_density_batch

    core_order_0 = compute_integrals(core_density_batch, integral_powers)
    core_scattering = scattering(core_density_batch)

    batch_order_0 = torch.stack((full_order_0, val_order_0, core_order_0),
                                dim=-1)
    batch_orders_1_and_2 = torch.stack((full_scattering, val_scattering,
                                        core_scattering), dim=-1)

    order_0.append(batch_order_0)
    orders_1_and_2.append(batch_orders_1_and_2)

order_0 = torch.cat(order_0, dim=0)
orders_1_and_2 = torch.cat(orders_1_and_2, dim=0)

n_molecules = order_0.shape[0]

np_order_0 = order_0.numpy().reshape((n_molecules, -1))
np_orders_1_and_2 = orders_1_and_2.numpy().reshape((n_molecules, -1))

basename = 'qm7_L_{}_J_{}_sigma_{}_MNO_{}_powers_{}.npy'.format(
        L, J, sigma, (M, N, O), integral_powers)
cachedir = get_cache_dir("qm7/experiments")
np.save(os.path.join(cachedir, 'order_0_' + basename),
        np_order_0)
np.save(os.path.join(cachedir, 'orders_1_and_2_' + basename),
        np_orders_1_and_2)

scattering_coef = np.concatenate([np_order_0, np_orders_1_and_2], axis=1)

qm7 = fetch_qm7()
target = qm7['energies']

n_folds = 5

n_datapoints = scattering_coef.shape[0]
P = np.random.permutation(n_datapoints).reshape((n_folds, -1))
cross_val_folds = []

for i_fold in range(n_folds):
    fold = (np.concatenate(P[np.arange(n_folds) != i_fold], axis=0), P[i_fold])
    cross_val_folds.append(fold)

alphas = 10.0 ** (-np.arange(0, 10))
for i, alpha in enumerate(alphas):
    regressor = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                       linear_model.Ridge(alpha=alpha))
    target_prediction = model_selection.cross_val_predict(regressor,
            X=scattering_coef, y=target,
                                                     cv=cross_val_folds)
    MAE = np.mean(np.abs(target_prediction - target))
    RMSE = np.sqrt(np.mean((target_prediction - target) ** 2))
    print('Ridge regression, alpha: {}, MAE: {}, RMSE: {}'.format(
        alpha, MAE, RMSE))
