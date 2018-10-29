import numpy as np
import time
import torch
import os

from sklearn import linear_model, model_selection, preprocessing, pipeline
from scattering.scattering3d import Scattering3D
from scattering.scattering3d.utils import compute_integrals, generate_weighted_sum_of_gaussians
from scattering.datasets import get_qm7_positions_and_charges, get_qm7_energies
from scattering.caching import get_cache_dir


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


def compute_qm7_solid_harmonic_scattering_coefficients(
        M=192, N=128, O=96, sigma=1.5, J=2, L=3,
        integral_powers=[0.5, 1., 2., 3.], batch_size=16):
    """
    Computes the scattering coefficients of the molecules of the
    QM7 database. Channels used are full charges, valence charges
    and core charges.

    Parameters
    ----------
    M, N, O: int
        number of points of the grid
    sigma : float
        width parameter of the Gaussian that represents a particle
    J: int
        maximal scale of the solid harmonic wavelets
    L: int
        maximal first order of the solid harmonic wavelets
    integral_powers: list of int
        powers for the integrals
    integral_powers: int
        size of the batch for computations

    Returns
    -------

    order_0: ndarray
        torch array containing the order 0 scattering coefficients
    order_1: ndarray
        torch array containing the order 1 scattering coefficients
    order_2: ndarray
        torch array containing the order 2 scattering coefficients
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

    scattering = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

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

        full_density_batch = generate_weighted_sum_of_gaussians(
                grid, pos_batch, full_batch, sigma, cuda=cuda)

        full_order_0 = compute_integrals(full_density_batch, integral_powers)
        full_order_1, full_order_2 = scattering(full_density_batch, order_2=True,
                                method='integral', integral_powers=integral_powers)

        full_order_0 = compute_integrals(full_density_batch, integral_powers)
        val_batch = valence_charges[start:end]
        val_density_batch = generate_weighted_sum_of_gaussians(
                grid, pos_batch, val_batch, sigma, cuda=cuda)
        val_order_0 = compute_integrals(val_density_batch, integral_powers)
        val_order_1, val_order_2 = scattering(val_density_batch, order_2=True,
                                method='integral', integral_powers=integral_powers)

        core_density_batch = full_density_batch - val_density_batch
        core_order_0 = compute_integrals(core_density_batch, integral_powers)
        core_order_1, core_order_2 = scattering(core_density_batch, order_2=True,
                                method='integral', integral_powers=integral_powers)

        order_0.append(
            torch.stack([full_order_0, val_order_0, core_order_0], dim=-1))
        order_1.append(
            torch.stack([full_order_1, val_order_1, core_order_1], dim=-1))
        order_2.append(
            torch.stack([full_order_2, val_order_2, core_order_2], dim=-1))

    order_0 = torch.cat(order_0, dim=0)
    order_1 = torch.cat(order_1, dim=0)
    order_2 = torch.cat(order_2, dim=0)

    return order_0, order_1, order_2

M, N, O, J, L = 192, 128, 96, 2, 2
integral_powers = [0.5,  1., 2., 3.]
sigma = 1.5

order_0, order_1, order_2 = compute_qm7_solid_harmonic_scattering_coefficients(
    M=M, N=N, O=O, J=J, L=L, integral_powers=integral_powers,
    sigma=sigma)

n_molecules = order_0.size(0)

np_order_0 = order_0.numpy().reshape((n_molecules, -1))
np_order_1 = order_1.numpy().reshape((n_molecules, -1))
np_order_2 = order_2.numpy().reshape((n_molecules, -1))

basename = 'qm7_L_{}_J_{}_sigma_{}_MNO_{}_powers_{}.npy'.format(
        L, J, sigma, (M, N, O), integral_powers)
cachedir = get_cache_dir("qm7/experiments")
np.save(os.path.join(cachedir, 'order_0_' + basename), np_order_0)
np.save(os.path.join(cachedir, 'order_1_' + basename), np_order_1)
np.save(os.path.join(cachedir, 'order_2_' + basename), np_order_2)

scattering_coef = np.concatenate([np_order_0, np_order_1, np_order_2], axis=1)
target = get_qm7_energies().numpy()

print('order 1 : {} coef, order 2 : {} coefs'.format(np_order_1.shape[1],
                                                    np_order_2.shape[1]))
evaluate_linear_regression(scattering_coef, target)
