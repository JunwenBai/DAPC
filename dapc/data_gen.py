# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# Salesforce Research (Junwen Bai, Weiran Wang)

import numpy as np
import torch
import scipy
from scipy.signal import resample

def gen_lorenz_system(T, integration_dt=0.005):
    """
    Period ~ 1 unit of time (total time is T)
    So make sure integration_dt << 1

    Known-to-be-good chaotic parameters
    See sussillo LFADS paper
    """
    rho = 28.0
    sigma = 10.0
    beta = 8 / 3.

    def dx_dt(state, t):
        x, y, z = state
        x_dot = sigma * (y - x)
        y_dot = x * (rho - z) - y
        z_dot = x * y - beta * z
        return (x_dot, y_dot, z_dot)

    x_0 = np.ones(3)
    t = np.arange(0, T, integration_dt)
    X = scipy.integrate.odeint(dx_dt, x_0, t)
    return X

def gen_lorenz_data(num_samples, normalize=True):
    integration_dt = 0.005
    data_dt = 0.025
    skipped_samples = 1000
    T = (num_samples + skipped_samples) * data_dt
    X = gen_lorenz_system(T, integration_dt)
    if normalize:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
    X_dwn = resample(X, num_samples + skipped_samples, axis=0)
    X_dwn = X_dwn[skipped_samples:, :]
    return X_dwn

def random_basis(N, D, rng):
    return scipy.stats.ortho_group.rvs(N, random_state=rng)[:, :D]

def median_subspace(N, D, rng, num_samples=5000, V_0=None):
    subspaces = np.zeros((num_samples, N, D)) # 5000*30*7
    angles = np.zeros((num_samples, min(D, V_0.shape[1]))) # 5000*3
    if V_0 is None:
        V_0 = np.eye(N)[:, :D]
    for i in range(num_samples):
        subspaces[i] = random_basis(N, D, rng)
        angles[i] = np.rad2deg(scipy.linalg.subspace_angles(V_0, subspaces[i]))
    median_angles = np.median(angles, axis=0)
    median_subspace_idx = np.argmin(np.sum((angles - median_angles)**2, axis=1))
    median_subspace = subspaces[median_subspace_idx]
    return median_subspace # 30*7

def gen_noise_cov(N, D, var, rng, V_noise=None):
    noise_spectrum = var * np.exp(-2 * np.arange(N) / D)
    if V_noise is None:
        V_noise = scipy.stats.ortho_group.rvs(N, random_state=rng)
    noise_cov = np.dot(V_noise, np.dot(np.diag(noise_spectrum), V_noise.T))
    return noise_cov

def gen_nonlinear_noisy_lorenz(N, T, snr=1., X_dynamics=None, noisy_model=None, V_dynamics=None, V_noise=None, seed=0, noise_dim=7, num_subspace_samples=5000):
    dynamics_var = np.max(scipy.linalg.eigvalsh(np.cov(X_dynamics.T)))
    X = noisy_model(torch.Tensor(X_dynamics)).detach().numpy()
    X_var = np.max(scipy.linalg.eigvalsh(np.cov(X.T)))
    X *= np.sqrt(dynamics_var/X_var)
    
    rng = np.random.RandomState(seed)
    noise_var = dynamics_var / snr

    if V_dynamics is None:
        if N == 3:
            V_dynamics = np.eye(3)
        else:
            V_dynamics = random_basis(N, 3, rng)

    if noise_dim == np.inf:
        noise_cov = np.eye(N) * noise_var
    else:
        # Generate a subspace with median principal angles w.r.t. dynamics subspace
        if V_noise is None:
            V_noise = median_subspace(N, noise_dim, rng, num_samples=num_subspace_samples, V_0=V_dynamics)
        # Extend V_noise to a basis for R^N
        if V_noise.shape[1] < N:
            V_noise_comp = scipy.linalg.orth(np.eye(N) - np.dot(V_noise, V_noise.T))
            V_noise = np.concatenate((V_noise, V_noise_comp), axis=1)
        # Add noise covariance
        noise_cov = gen_noise_cov(N, noise_dim, noise_var, rng, V_noise=V_noise)

    X_samples = X + rng.multivariate_normal(mean=np.zeros(N), cov=noise_cov, size=len(X_dynamics))
    return X, X_samples
