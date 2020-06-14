import logging, time
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.signal.windows import hann

import torch
import torch.nn.functional as F

from .solver import DNN
from .utils import calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats

__all__ = ['DynamicalComponentsAnalysis']

logging.basicConfig()

def ortho_reg_fn(V, ortho_lambda):
    """Regularization term which encourages the basis vectors in the
    columns of V to be orthonormal.
    Parameters
    ----------
    V : np.ndarray, shape (N, d)
        Matrix whose columns are basis vectors.
    ortho_lambda : float
        Regularization hyperparameter.
    Returns
    -------
    reg_val : float
        Value of regularization function.
    """

    use_torch = isinstance(V, torch.Tensor)
    d = V.shape[1]

    if use_torch:
        reg_val = ortho_lambda * torch.sum((torch.mm(V.t(), V) -
                                            torch.eye(d, device=V.device, dtype=V.dtype))**2)
    else:
        reg_val = ortho_lambda * np.sum((np.dot(V.T, V) - np.eye(d))**2)

    return reg_val

def smooth_reg_Y(Y, smooth_lambda=1.):
    d = Y.shape[1]
    Y_flat = Y.flatten()
    smooth_reg = F.l1_loss(Y_flat[d:], Y_flat[:-d]) * smooth_lambda
    return smooth_reg

def ortho_reg_Y(Y, ortho_lambda=1.):
    d = Y.shape[1]
    I = torch.eye(d)
    ones = torch.ones(d, d)

    reg_val = ortho_lambda * torch.sum(((torch.mm(Y.t(), Y) - I))**2)

    return reg_val

class DynamicalComponentsAnalysis(object):
    """Dynamical Components Analysis.

    Runs DCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the complexity, as defined by the Gaussian
    Predictive Information (PI) of the d-dimensional dynamics over windows of length T.

    Parameters
    ----------
    d : int
        Number of basis vectors onto which the data X are projected.
    T : int
        Size of time windows across which to compute mutual information. Total window length will be
        `2 * T`. When fitting a model, the length of the shortest timeseries must be greater than
        `2 * T` and for good performance should be much greater than `2 * T`.
    init : str
        Options: "random_ortho", "random", or "PCA"
        Method for initializing the projection matrix.
    n_init : int
        Number of random restarts. Default is 1.
    tol : float
        Tolerance for stopping optimization. Default is 1e-6.
    ortho_lambda : float
        Coefficient on term that keeps V close to orthonormal.
    verbose : bool
        Verbosity during optimization.
    use_scipy : bool
        Whether to use SciPy or Pytorch L-BFGS-B. Default is True. Pytorch is not well tested.
    block_toeplitz : bool
        If True, uses the block-Toeplitz logdet algorithm which is typically faster and less
        memory intensive on cpu for `T >~ 10` and `d >~ 40`.
    chunk_cov_estimate : None or int
        If `None`, cov is estimated from entire time series. If an `int`, cov is estimated
        by chunking up time series and averaging covariances from chucks. This can use less memory
        and be faster for long timeseries. Requires that the length of the shortest timeseries
        in the batch is longer than `2 * T * chunk_cov_estimate`.
    device : str
        What device to run the computation on in Pytorch.
    dtype : pytorch.dtype
        What dtype to use for computation.
    """
    def __init__(self, d=None, T=None, init="random_ortho", n_init=1, tol=1e-6,
                 ortho_lambda=10., smooth_lambda=0., verbose=False, use_scipy=True, block_toeplitz=None,
                 chunk_cov_estimate=None, max_epochs=2000, dropout=0.5, device="cpu", dtype=torch.float64, rng_or_seed=None):
        self.d = d
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.smooth_lambda = smooth_lambda
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.dropout = dropout
        self._logger = logging.getLogger('d-DCA')
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)
        self.device = device
        self.dtype = dtype
        self.use_scipy = use_scipy
        if block_toeplitz is None:
            try:
                if d > 40 and T > 10:
                    self.block_toeplitz = True
                else:
                    self.block_toeplitz = False
            except TypeError:
                self.block_toeplitz = False
        else:
            self.block_toeplitz = block_toeplitz
        self.chunk_cov_estimate = chunk_cov_estimate
        self.cross_covs = None
        self.coef_ = None
        self.mean_ = None
        if rng_or_seed is None:
            self.rng = np.random
        elif isinstance(rng_or_seed, np.random.RandomState):
            self.rng = rng_or_seed
        else:
            self.rng = np.random.RandomState(rng_or_seed)

    def estimate_cross_covariance(self, X, T=None, regularization=None, reg_ops=None):
        """Estimate the cross covariance matrix from data.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        T : int
            T for PI calculation (optional.)
        regularization : str
            Whether to regularize cross covariance estimation.
        reg_ops : dict
            Options for cross covariance regularization.
        """
        if T is None:
            T = self.T
        else:
            self.T = T
        start = time.time()
        self._logger.info('Starting cross covariance estimate.')
        if isinstance(X, list) or X.ndim == 3:
            self.mean_ = np.concatenate(X).mean(axis=0, keepdims=True)
        else:
            self.mean_ = X.mean(axis=0, keepdims=True)

        cross_covs = calc_cross_cov_mats_from_data(X, 2 * self.T, mean=self.mean_,
                                                   chunks=self.chunk_cov_estimate,
                                                   regularization=regularization,
                                                   reg_ops=reg_ops)
        
        self.cross_covs = cross_covs
        delta_time = round((time.time() - start) / 60., 1)
        self._logger.info('Cross covariance estimate took {:0.1f} minutes.'.format(delta_time))

        return self.cross_covs

    def build_nn_loss(self, Y, ortho_lambda=1., smooth_lambda=1.):
        c = self.estimate_cross_covariance(Y) # cross covariance: 2T * d * d 
        loss = -calc_pi_from_cross_cov_mats(c) # compute negative pi from the cross covariance
        ortho_reg = ortho_reg_Y(Y, ortho_lambda) # orthognal regularization
        #smooth_reg = smooth_reg_Y(Y, smooth_lambda)
        return loss + ortho_reg, loss

    def _fit_projection(self, X_train, X_val, Y_val_gt, writer, d=None, record_V=False):
        if d is None:
            d = self.d
        if d < 1:
            raise ValueError

        N = X_train.shape[1]

        model = DNN(N, d, dropout=self.dropout) # Dim reduction NN
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
        X_train = torch.Tensor(X_train)
        X_val = torch.Tensor(X_val)
        batch_size = 1024
        n_batch = int(len(X_train)/batch_size)
        for epoch in range(self.max_epochs):
            tot_pi = 0.
            for i in range(n_batch):
                model.train()
                optimizer.zero_grad()
                Y_train = model(X_train[i*batch_size: (i+1)*batch_size])
                loss, neg_pi = self.build_nn_loss(Y_train, self.ortho_lambda, self.smooth_lambda) # build loss and pi for the latent low-dim representations
                tot_pi += -neg_pi
                loss.backward()
                optimizer.step()

            writer.add_scalar('train/train dDCA PI', -tot_pi/n_batch, epoch)
            final_pi = -tot_pi/n_batch
            if epoch % 50 == 0:
                print(epoch, ":", -loss.item())

            model.eval()
            Y_val = model(X_val)
            val_loss, neg_val_pi = self.build_nn_loss(Y_val, self.ortho_lambda, self.smooth_lambda) # evaluate loss and pi for the validation set
            writer.add_scalar('val/val dDCA PI', -neg_val_pi, epoch)

            gt_loss, neg_gt_pi = self.build_nn_loss(Y_val_gt, self.ortho_lambda, self.smooth_lambda) # evaluate the ground-truth pi
            writer.add_scalar('gt/gt dDCA PI', -neg_gt_pi, epoch)

        self.model = model

        return final_pi

    def fit(self, X_train, X_val, Y_val_gt, writer, d=None, T=None, regularization=None, reg_ops=None, n_init=None):
        """Estimate the cross covariance matrix and fit the projection matrix.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional.)
        regularization : str
            Whether to regularize cross covariance estimation.
        reg_ops : dict
            Options for cross covariance regularization.
        n_init : int
            Number of random restarts (optional.)
        """
        if n_init is None:
            n_init = self.n_init
        
        self._fit_projection(X_train, X_val, Y_val_gt, writer, d=d) # fit X_train and eval on X_val
        return self

