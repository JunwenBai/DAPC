import logging, time
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.signal.windows import hann

import torch
import torch.nn.functional as F

from .solver import DNN
from .cov_util import (calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats,
                       form_lag_matrix, calc_pi_from_cross_cov_mats_block_toeplitz)

__all__ = ['DynamicalComponentsAnalysis',
           'DynamicalComponentsAnalysisFFT',
           'DynamicalComponentsAnalysisKNN',
           'ortho_reg_fn',
           'build_loss',
           'init_coef']


logging.basicConfig()


def ortho_reg_Y(Y, ortho_lambda):
    d = Y.shape[1]
    I = torch.eye(d)
    ones = torch.ones(d, d)

    reg_val = ortho_lambda * torch.sum(((torch.mm(Y.t(), Y) - I))**2)

    return reg_val


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


def build_loss(cross_cov_mats, d, ortho_lambda=1., block_toeplitz=False):
    """Constructs a loss function which gives the (negative) predictive
    information in the projection of multidimensional timeseries data X onto a
    d-dimensional basis, where predictive information is computed using a
    stationary Gaussian process approximation.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional time series data from which the
        mutual information is computed.
    d: int
        Number of basis vectors onto which the data X are projected.
    ortho_lambda : float
        Regularization hyperparameter.
    Returns
    -------
    loss : function
       Loss function which accepts a (flattened) N-by-d matrix, whose
       columns are basis vectors, and outputs the negative predictive information
       corresponding to that projection (plus regularization term).
    """
    N = cross_cov_mats.shape[1]

    '''print("cross_cov_mats:", cross_cov_mats.shape)
    print("N:", N)
    print("d:", d)
    print("ortho_lambda:", ortho_lambda)
    print("blk_toe:", block_toeplitz)'''

    if block_toeplitz:
        def loss(V_flat):
            V = V_flat.reshape(N, d)
            reg_val = ortho_reg_fn(V, ortho_lambda)
            return -calc_pi_from_cross_cov_mats_block_toeplitz(cross_cov_mats, V) + reg_val
    else:
        def loss(V_flat):
            V = V_flat.reshape(N, d)
            reg_val = ortho_reg_fn(V, ortho_lambda)
            return -calc_pi_from_cross_cov_mats(cross_cov_mats, V) + reg_val

    return loss

    


class ObjectiveWrapper(object):
    """Helper object to cache gradient computation for minimization.

    Parameters
    ----------
    f_params : callable
        Function to calculate the loss as a function of the parameters.
    """
    def __init__(self, f_params):
        self.common_computations = None
        self.params = None
        self.f_params = f_params
        self.n_f = 0
        self.n_g = 0
        self.n_c = 0

    def core_computations(self, *args, **kwargs):
        """Calculate the part of the computation that is common to computing
        the loss and the gradient.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
        params = args[0]
        if not np.array_equal(params, self.params):
            self.n_c += 1
            self.common_computations = self.f_params(*args, **kwargs)
            self.params = params.copy()
        return self.common_computations

    def func(self, *args):
        """Calculate and return the loss.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
        self.n_f += 1
        loss, _ = self.core_computations(*args)
        return loss.detach().cpu().numpy().astype(float)

    def grad(self, *args):
        """Calculate and return the gradient of the loss.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
        self.n_g += 1
        loss, params_torch = self.core_computations(*args)
        loss.backward(retain_graph=True)
        grad = params_torch.grad
        return grad.detach().cpu().numpy().astype(float)


def init_coef(N, d, rng, init):
    """Initialize a projection coefficent matrix.

    Parameters
    ----------
    N : int
        Original dimensionality.
    d : int
        Projected dimensionality.
    rng : np.random.RandomState
        Random state for generation.
    init : str or ndarray
        Initialization type.
    """
    if type(init) == str:
        if init == "random":
            V_init = rng.normal(0, 1, (N, d))
        elif init == "random_ortho":
            V_init = scipy.stats.ortho_group.rvs(N, random_state=rng)[:, :d]
        elif init == "uniform":
            V_init = np.ones((N, d)) / np.sqrt(N)
            V_init = V_init + rng.normal(0, 1e-3, V_init.shape)
        else:
            raise ValueError
    elif isinstance(init, np.ndarray):
        V_init = init.copy()
    else:
        raise ValueError
    #print("V_init:", V_init.shape)
    V_init /= np.linalg.norm(V_init, axis=0, keepdims=True)
    return V_init


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
                 ortho_lambda=10., verbose=False, use_scipy=True, block_toeplitz=None,
                 chunk_cov_estimate=None, device="cpu", dtype=torch.float64, rng_or_seed=None):
        self.d = d
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose = verbose
        self._logger = logging.getLogger('DCA')
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

    def fit_projection(self, d=None, n_init=None):
        """Fit the projection matrix.

        Parameters
        ----------
        d : int
            Dimensionality of the projection (optional.)
        n_init : int
            Number of random restarts (optional.)
        """
        if n_init is None:
            n_init = self.n_init
        pis = []
        coefs = []
        for ii in range(n_init):
            start = time.time()
            self._logger.info('Starting projection fig {} of {}.'.format(ii + 1, n_init))
            coef, pi = self._fit_projection(d=d)
            #print("pi:", pi)
            #print("coef:", coef.shape)
            delta_time = round((time.time() - start) / 60., 1)
            self._logger.info('Projection fit {} of {} took {:0.1f} minutes.'.format(ii + 1,
                                                                                     n_init,
                                                                                     delta_time))
            pis.append(pi)
            coefs.append(coef)
        idx = np.argmax(pis)
        self.coef_ = coefs[idx]

    def build_nn_loss(self, Y, ortho_lambda=1.):
        c = self.estimate_cross_covariance(Y) # cross covariance: 2T * d * d 
        loss = -calc_pi_from_cross_cov_mats(c) # compute negative pi from the cross covariance
        reg = ortho_reg_Y(Y, ortho_lambda) # orthognal regularization
        return loss + reg, loss

    def _fit_projection(self, X_train, X_val, Y_val_gt, writer, d=None, record_V=False):
        if d is None:
            d = self.d
        if d < 1:
            raise ValueError

        N = X_train.shape[1]

        model = DNN(N, d) # Dim reduction NN
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
        X_train = torch.Tensor(X_train) 
        X_val = torch.Tensor(X_val) 
        batch_size = 1024
        n_batch = int(len(X_train)/batch_size)
        for epoch in range(501):
            tot_pi = 0.
            for i in range(n_batch):
                model.train()
                optimizer.zero_grad()
                Y_train = model(X_train[i*batch_size: (i+1)*batch_size])
                loss, neg_pi = self.build_nn_loss(Y_train, self.ortho_lambda) # build loss and pi for the latent low-dim representations
                tot_pi += -neg_pi
                loss.backward()
                optimizer.step()

            writer.add_scalar('train/train dDCA PI', -tot_pi/n_batch, epoch)
            final_pi = -tot_pi/n_batch
            if epoch % 50 == 0:
                print(epoch, ":", -loss.item())

            model.eval()
            Y_val = model(X_val)
            val_loss, neg_val_pi = self.build_nn_loss(Y_val, self.ortho_lambda) # evaluate loss and pi for the validation set
            writer.add_scalar('val/val dDCA PI', -neg_val_pi, epoch)

            gt_loss, neg_gt_pi = self.build_nn_loss(Y_val_gt, self.ortho_lambda) # evaluate the ground-truth pi
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

    def transform(self, X):
        """Project the data onto the DCA components after removing the training
        mean.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        """
        if isinstance(X, list):
            y = [(Xi - self.mean_).dot(self.coef_) for Xi in X]
        elif X.ndim == 3:
            y = np.stack([(Xi - self.mean_).dot(self.coef_) for Xi in X])
        else:
            y = (X - self.mean_).dot(self.coef_)
        return y

    def fit_transform(self, X, d=None, T=None, regularization=None,
                      reg_ops=None, n_init=None):
        """Estimate the cross covariance matrix and fit the projection matrix. Then
        project the data onto the DCA components.

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
        self.fit(X, d=d, T=T, regularization=regularization, reg_ops=reg_ops, n_init=n_init)
        return self.transform(X)

    def score(self, X=None):
        """Calculate the PI of data for the DCA projection.

        Parameters
        ----------
        X : ndarray or list
            Optional. If X is none, calculate PI from the training data.
            If X is given, calcuate the PI of X for the learned projections.
        """
        if X is None:
            cross_covs = self.cross_covs
        else:
            cross_covs = calc_cross_cov_mats_from_data(X, T=self.T)
        if self.block_toeplitz:
            return calc_pi_from_cross_cov_mats_block_toeplitz(cross_covs, self.coef_)
        else:
            return calc_pi_from_cross_cov_mats(cross_covs, self.coef_)


def make_cepts2(X, T_pi):
    """Calculate the squared real cepstral coefficents."""
    Y = F.unfold(X, kernel_size=[T_pi, 1], stride=T_pi)
    Y = torch.transpose(Y, 1, 2)

    # Compute the power spectral density
    window = torch.Tensor(hann(Y.shape[-1])[np.newaxis, np.newaxis]).type(Y.dtype)
    Yf = torch.rfft(Y * window, 1, onesided=True)
    spect = Yf[:, :, :, 0]**2 + Yf[:, :, :, 1]**2
    spect = spect.mean(dim=1)
    spect = torch.cat([torch.flip(spect[:, 1:], dims=(1,)), spect], dim=1)

    # Log of the DFT of the autocorrelation
    logspect = torch.log(spect) - np.log(float(Y.shape[-1]))

    # Compute squared cepstral coefs (b_k^2)
    cepts = torch.rfft(logspect, 1, onesided=True) / float(Y.shape[-1])
    cepts = torch.sqrt(cepts[:, :, 0]**2 + cepts[:, :, 1]**2)
    return cepts**2

