import numpy as np
import torch
import scipy as sp
import pdb

def torch_toeplitzify(cov, T, d, symmetrize=True):
    cov_toep = torch.zeros(T * d, T * d, device=cov.device)
    for delta_t in range(T):
        to_avg_lower = torch.zeros(T - delta_t, d, d)
        to_avg_upper = torch.zeros(T - delta_t, d, d)
        for i in range(T - delta_t):
            to_avg_lower[i] = cov[(delta_t + i) * d:(delta_t + i + 1) * d, i * d:(i + 1) * d]
            to_avg_upper[i] = cov[i * d:(i + 1) * d, (delta_t + i) * d:(delta_t + i + 1) * d]
        avg_lower = torch.mean(to_avg_lower, axis=0, device=cov.device)
        avg_upper = torch.mean(to_avg_upper, axis=0, device=cov.device)
        if symmetrize:
            avg_lower = 0.5 * (avg_lower + avg_upper.T)
            avg_upper = avg_lower.T
        for i in range(T - delta_t):
            cov_toep[(delta_t + i) * d:(delta_t + i + 1) * d, i * d:(i + 1) * d] = avg_lower
            cov_toep[i * d:(i + 1) * d, (delta_t + i) * d:(delta_t + i + 1) * d] = avg_upper
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
    logdet_T_pi = torch.logdet(cov_T_pi)
    logdet_2T_pi = torch.logdet(cov_2_T_pi)

    PI = logdet_T_pi - .5 * logdet_2T_pi
    return PI


def calc_cov_from_data(xs_pad, src_mask, T, toeplitzify=True, reg=0.0):
    """Compute the TN-by-TN cross-covariance matrix, where d is the data dimensionality,
    for each time lag up to T-1.

    Parameters
    ----------
    X : shape (batch, maxlen, d)
        The d-dimensional time series data from which the covariance matrices are computed.
    src_mask: shape (batch, maxlen), type: bool
        The length of each sequence in X
    T : int
        The number of time lags.

    Returns
    -------
    cov_est : shape (TN, TN), float
        Covariance matrices for length T time steps. cov_est[t1, t2] is the cross-covariance between
        X(t1) and X(t2), where X(t) is an d-dimensional vector.
    """

    B = xs_pad.size(0)
    maxlen = xs_pad.size(1)
    d = xs_pad.size(2)

    if torch.min(src_mask.sum(1)) <= T:
        raise ValueError('T must be shorter than the length of the shortest ' +
                         'time series. If you are using the DCA model, 2 * DCA.T must be ' +
                         'shorter than the shortest time series.')

    """
    # Weiran: since we are going to remove mean later, this step is omitted.
    mask_float = src_mask.float().flatten().unsqueeze(1)
    mean = torch.sum(torch.mul(xs_pad.view([B*maxlen, d]), mask_float), 0, keepdim=True) / torch.sum(mask_float)
    xs_pad = xs_pad - mean.unsqueeze(0)
    """

    # Extracts sliding local blocks.
    xs_with_lags = xs_pad.view([B, maxlen*d]).unfold(1, T*d, d)
    # Find the valid concat frames.
    mask_with_lags = src_mask.unfold(1, T, 1).all(dim=2)
    mask_float = mask_with_lags.float().flatten().unsqueeze(1)
    xs_with_lags = torch.reshape(xs_with_lags, [-1, T*d])
    xs_with_lags_mean = torch.sum(torch.mul(xs_with_lags, mask_float), 0, keepdim=True) / torch.sum(mask_float)
    # Remove mean for concat frames.
    xs_with_lags = xs_with_lags - xs_with_lags_mean

    cov_est = torch.mul(xs_with_lags, mask_float).t().matmul(xs_with_lags) / torch.sum(mask_float)
    if toeplitzify:
        cov_est = torch_toeplitzify(cov_est, T, d)

    # rectify_spectrum(cov_est, verbose=False)
    if reg>0:
        cov_est = cov_est + reg * torch.eye(T*d, T*d, device=cov_est.device)

    return cov_est

