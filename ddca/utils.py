import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy as sp
from scipy.signal import resample


def torch_toeplitzify(cov, T, N, symmetrize=True):
    cov_toep = torch.zeros(T * N, T * N)
    for delta_t in range(T):
        to_avg_lower = torch.zeros(T - delta_t, N, N)
        to_avg_upper = torch.zeros(T - delta_t, N, N)
        for i in range(T - delta_t):
            to_avg_lower[i] = cov[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N]
            to_avg_upper[i] = cov[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N]
        avg_lower = torch.mean(to_avg_lower, axis=0)
        avg_upper = torch.mean(to_avg_upper, axis=0)
        if symmetrize:
            avg_lower = 0.5 * (avg_lower + avg_upper.T)
            avg_upper = avg_lower.T
        for i in range(T - delta_t):
            cov_toep[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N] = avg_lower
            cov_toep[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N] = avg_upper
    return cov_toep


def rectify_spectrum(cov, epsilon=1e-6, verbose=False):
    """Rectify the spectrum of a covariance matrix.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix
    epsilon : float
        Minimum eigenvalue for the rectified spectrum.
    verbose : bool
        Whethere to print when the spectrum needs to be rectified.
    """
    min_eig = np.min(sp.linalg.eigvalsh(cov.detach().cpu().numpy()))
    if min_eig < 0:
        cov += (-min_eig + epsilon) * torch.eye(cov.shape[0])
        if verbose:
            print("Warning: non-PSD matrix (had to increase eigenvalues)")


def calc_cross_cov_mats_from_cov(cov, T, N):
    """Calculates T N-by-N cross-covariance matrices given
    a N*T-by-N*T spatiotemporal covariance matrix by
    averaging over off-diagonal cross-covariance blocks with
    constant `|t1-t2|`.
    Parameters
    ----------
    N : int
        Numbner of spatial dimensions.
    T: int
        Number of time-lags.
    cov : np.ndarray, shape (N*T, N*T)
        Spatiotemporal covariance matrix.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices.
    """

    cross_cov_mats = torch.zeros((T, N, N))

    for delta_t in range(T):
        to_avg_lower = torch.zeros((T - delta_t, N, N))
        to_avg_upper = torch.zeros((T - delta_t, N, N))

        for i in range(T - delta_t):
            to_avg_lower[i, :, :] = cov[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N]
            to_avg_upper[i, :, :] = cov[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N]

        avg_lower = to_avg_lower.mean(axis=0)
        avg_upper = to_avg_upper.mean(axis=0)

        cross_cov_mats[delta_t, :, :] = 0.5 * (avg_lower + avg_upper.t())

    return cross_cov_mats


def calc_cross_cov_mats_from_data(X, T, mean=None, chunks=None, regularization=None, reg_ops=None,
                                  stride_tricks=True):
    """Compute the N-by-N cross-covariance matrix, where N is the data dimensionality,
    for each time lag up to T-1.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The N-dimensional time series data from which the cross-covariance
        matrices are computed.
    T : int
        The number of time lags.
    chunks : int
        Number of chunks to break the data into when calculating the lagged cross
        covariance. More chunks will mean less memory used
    regularization : string
        Regularization method for computing the spatiotemporal covariance matrix.
    reg_ops : dict
        Paramters for regularization.
    stride_tricks : bool
        Whether to use numpy stride tricks in form_lag_matrix. True will use less
        memory for large T.

    Returns
    -------
    cross_cov_mats : np.ndarray, shape (T, N, N), float
        Cross-covariance matrices. cross_cov_mats[dt] is the cross-covariance between
        X(t) and X(t+dt), where X(t) is an N-dimensional vector.
    """
    if reg_ops is None:
        reg_ops = dict()
    stride = reg_ops.get('stride', 1)
        
    if len(X) <= T:
        raise ValueError('T must be shorter than the length of the shortest ' +
                         'timeseries. If you are using the DCA model, 2 * DCA.T must be ' +
                         'shorter than the shortest timeseries.')
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    X = X - mean
    N = X.shape[-1]
    num_samples = X.shape[0]
    
    if chunks is None:
        X_with_lags = X.flatten().unfold(0, N*T, N)
        X_with_lags_mean = X_with_lags.mean(axis=0, keepdims=True)
        cov_est = (X_with_lags - X_with_lags_mean).t().matmul(X_with_lags - X_with_lags_mean) / (num_samples-1.)

    if regularization is None:
        cov_est = torch_toeplitzify(cov_est, T, N)

    rectify_spectrum(cov_est, verbose=False)
    cross_cov_mats = calc_cross_cov_mats_from_cov(cov_est, T, N)
    
    return cross_cov_mats


def calc_cov_from_cross_cov_mats(cross_cov_mats):
    """Calculates the N*T-by-N*T spatiotemporal covariance matrix based on
    T N-by-N cross-covariance matrices.

    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.

    Returns
    -------
    cov : np.ndarray, shape (N*T, N*T)
        Big covariance matrix, stationary in time by construction.
    """

    N = cross_cov_mats.shape[1]
    T = len(cross_cov_mats)

    cross_cov_mats_repeated = []
    for i in range(T):
        for j in range(T):
            if i > j:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i - j)])
            else:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i - j)].t())

    cov_tensor = torch.reshape(torch.stack(cross_cov_mats_repeated), (T, T, N, N))
    cov = torch.cat([torch.cat([cov_ii_jj for cov_ii_jj in cov_ii], dim=1)
                     for cov_ii in cov_tensor])
    return cov


def calc_pi_from_cov(cov_2_T_pi):
    """Calculates the mutual information ("predictive information"
    or "PI") between variables  {1,...,T_pi} and {T_pi+1,...,2*T_pi}, which
    are jointly Gaussian with covariance matrix cov_2_T_pi.

    Parameters
    ----------
    cov_2_T_pi : np.ndarray, shape (2*T_pi, 2*T_pi)
        Covariance matrix.

    Returns
    -------
    PI : float
        Mutual information in nats.
    """
    T_pi = cov_2_T_pi.shape[0] // 2

    cov_T_pi = cov_2_T_pi[:T_pi, :T_pi]
    logdet_T_pi = torch.slogdet(cov_T_pi)[1]
    logdet_2T_pi = torch.slogdet(cov_2_T_pi)[1]

    PI = logdet_T_pi - .5 * logdet_2T_pi
    return PI


def calc_pi_from_cross_cov_mats(cross_cov_mats, proj=None):
    """Calculates predictive information for a spatiotemporal Gaussian
    process with T-1 N-by-N cross-covariance matrices.

    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    proj: np.ndarray, shape (N, d), optional
        If provided, the N-dimensional data are projected onto a d-dimensional
        basis given by the columns of proj. Then, the mutual information is
        computed for this d-dimensional timeseries.

    Returns
    -------
    PI : float
        Mutual information in nats.
    """
    
    cross_cov_mats_proj = cross_cov_mats

    cov_2_T_pi = calc_cov_from_cross_cov_mats(cross_cov_mats_proj)
    PI = calc_pi_from_cov(cov_2_T_pi)

    return PI

