"""
3D scattering quantum chemistry regression
==========================================
This uses the 3D scattering on a standard dataset.
"""

import numpy as np
import time
import torch
import os

from sklearn import linear_model, model_selection, preprocessing, pipeline
from kymatio.scattering3d import HarmonicScattering3D
from kymatio.scattering3d.utils import compute_integrals, generate_weighted_sum_of_gaussians
from kymatio.datasets import fetch_qm7
from kymatio.caching import get_cache_dir
from scipy.spatial.distance import pdist


def evaluate_linear_regression(X, y, n_folds=5):
    """
    Evaluates linear ridge regression predictions of y using X.

    Parameters
    ----------
    X:  numpy array
        input features, shape (N, D)
    y: numpy array
        target value, shape (N, 1)

    """
    n_datapoints = X.shape[0]
    P = np.random.permutation(n_datapoints).reshape((n_folds, -1))
    cross_val_folds = []

    for i_fold in range(n_folds):
        fold = (np.concatenate(P[np.arange(n_folds) != i_fold], axis=0), P[i_fold])
        cross_val_folds.append(fold)

    alphas = 10.**(-np.arange(0, 10))
    for i, alpha in enumerate(alphas):
        regressor = pipeline.make_pipeline(
            preprocessing.StandardScaler(), linear_model.Ridge(alpha=alpha))
        y_prediction = model_selection.cross_val_predict(
            regressor, X=X, y=y, cv=cross_val_folds)
        MAE = np.mean(np.abs(y_prediction - y))
        RMSE = np.sqrt(np.mean((y_prediction - y)**2))
        print('Ridge regression, alpha: {}, MAE: {}, RMSE: {}'.format(
            alpha, MAE, RMSE))


def get_valence(charges):
    """
        Returns the number valence electrons of a particle given the
        nuclear charge.

        Parameters
        ----------
        charges: numpy array
            array containing the nuclear charges, arbitrary size

        Returns
        -------
        valence_charges : numpy array
            same size as the input
    """
    return (
        charges * (charges <= 2) +
        (charges - 2) * np.logical_and(charges > 2, charges <= 10) +
        (charges - 10) * np.logical_and(charges > 10, charges <= 18))


def get_qm7_energies():
    """
        Loads the energies of the molecules of the QM7 dataset.

        Returns
        -------
        energies: numpy array
            array containing the energies of the molecules
    """
    qm7 = fetch_qm7()
    return qm7['energies']



def get_qm7_positions_and_charges(sigma, overlapping_precision=1e-1):
    """
        Loads the positions and charges of the molecules of the QM7 dataset.
        QM7 is a dataset of 7165 organic molecules with up to 7 non-hydrogen
        atoms, whose energies were computed with a quantun chemistry
        computational method named Density Functional Theory.
        This dataset has been made available to train machine learning models
        to predict these energies.

        Parameters
        ----------
        sigma : float
            width parameter of the Gaussian that represents a particle

        overlapping_precision : float, optional
            affects the scaling of the positions. The positions are re-scaled
            such that two Gaussian functions of width sigma centerd at the qm7
            positions overlapp with amplitude <= the overlapping_precision

        Returns
        -------
        positions, charges, valence_charges: torch arrays
            array containing the positions, charges and valence charges
            of the QM7 database molecules
    """
    qm7 = fetch_qm7(align=True)
    positions = qm7['positions']
    charges = qm7['charges'].astype('float32')
    valence_charges = get_valence(charges)

    # normalize positions
    min_dist = np.inf
    for i in range(positions.shape[0]):
        n_atoms = np.sum(charges[i] != 0)
        pos = positions[i, :n_atoms, :]
        min_dist = min(min_dist, pdist(pos).min())
    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
    positions = positions * delta / min_dist

    return (torch.from_numpy(positions),
            torch.from_numpy(charges),
            torch.from_numpy(valence_charges))


def compute_qm7_solid_harmonic_scattering_coefficients(
        M=192, N=128, O=96, sigma=2., J=2, L=3,
        integral_powers=(0.5, 1., 2., 3.), batch_size=16):
    """
        Computes the scattering coefficients of the molecules of the
        QM7 database. Channels used are full charges, valence charges
        and core charges. Linear regression of the qm7 energies with
        the given values gives MAE 2.75, RMSE 4.18 (kcal.mol-1).

        Parameters
        ----------
        M, N, O: int
            dimensions of the numerical grid
        sigma : float
            width parameter of the Gaussian that represents a particle
        J: int
            maximal scale of the solid harmonic wavelets
        L: int
            maximal first order of the solid harmonic wavelets
        integral_powers: list of int
            powers for the integrals
        batch_size: int
            size of the batch for computations

        Returns
        -------
        order_0: torch tensor
            array containing zeroth-order scattering coefficients
        orders_1_and_2: torch tensor
            array containing first- and second-order scattering coefficients
    """
    cuda = torch.cuda.is_available()
    grid = torch.from_numpy(
        np.fft.ifftshift(
            np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'),
            axes=(1, 2, 3)))
    pos, full_charges, valence_charges = get_qm7_positions_and_charges(sigma)

    if cuda:
        grid = grid.cuda()
        pos = pos.cuda()
        full_charges = full_charges.cuda()
        valence_charges = valence_charges.cuda()

    n_molecules = pos.size(0)
    n_batches = np.ceil(n_molecules / batch_size).astype(int)

    scattering = HarmonicScattering3D(J=J, shape=(M, N, O), L=L, sigma_0=sigma)

    order_0, order_1, order_2 = [], [], []
    print('Computing solid harmonic scattering coefficients of {} molecules '
        'of QM7 database on {}'.format(pos.size(0), 'GPU' if cuda else 'CPU'))
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
                        int(((n_batches - i - 1) * dt) % 60)), end='\r')
        else:
            print("Iteration {} ETA: {}".format(i + 1,'-'),end='\r')
        last_time = this_time
        time.sleep(1)

        start, end = i * batch_size, min((i + 1) * batch_size, n_molecules)

        pos_batch = pos[start:end]
        full_batch = full_charges[start:end]
        val_batch = valence_charges[start:end]

        full_density_batch = generate_weighted_sum_of_gaussians(
                grid, pos_batch, full_batch, sigma, cuda=cuda)
        full_order_0 = compute_integrals(full_density_batch, integral_powers)
        scattering.max_order = 2
        scattering.method = 'integral'
        scattering.integral_powers = integral_powers
        full_scattering = scattering(full_density_batch)

        val_density_batch = generate_weighted_sum_of_gaussians(
                grid, pos_batch, val_batch, sigma, cuda=cuda)
        val_order_0 = compute_integrals(val_density_batch, integral_powers)
        val_scattering= scattering(val_density_batch)

        core_density_batch = full_density_batch - val_density_batch
        core_order_0 = compute_integrals(core_density_batch, integral_powers)
        core_scattering = scattering(core_density_batch)


        order_0.append(
            torch.stack([full_order_0, val_order_0, core_order_0], dim=-1))
        orders_1_and_2.append(
            torch.stack(
                [full_scattering, val_scattering, core_scattering], dim=-1))

    order_0 = torch.cat(order_0, dim=0)
    orders_1_and_2 = torch.cat(orders_1_and_2, dim=0)

    return order_0, orders_1_and_2

M, N, O, J, L = 192, 128, 96, 2, 3
integral_powers = [0.5,  1., 2., 3.]
sigma = 2.

order_0, orders_1_and_2 = compute_qm7_solid_harmonic_scattering_coefficients(
    M=M, N=N, O=O, J=J, L=L, integral_powers=integral_powers,
    sigma=sigma, batch_size=8)

n_molecules = order_0.size(0)

np_order_0 = order_0.numpy().reshape((n_molecules, -1))
np_orders_1_and_2 = orders_1_and_2.numpy().reshape((n_molecules, -1))

basename = 'qm7_L_{}_J_{}_sigma_{}_MNO_{}_powers_{}.npy'.format(
        L, J, sigma, (M, N, O), integral_powers)
cachedir = get_cache_dir("qm7/experiments")
np.save(os.path.join(cachedir, 'order_0_' + basename), np_order_0)
np.save(os.path.join(
    cachedir, 'orders_1_and_2_' + basename), np_orders_1_and_2)

scattering_coef = np.concatenate([np_order_0, np_orders_1_and_2], axis=1)
target = get_qm7_energies()

evaluate_linear_regression(scattering_coef, target)
