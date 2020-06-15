import logging, time
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.signal.windows import hann

import torch
import torch.nn.functional as F

from .solver import DNN, RNN
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
    num_samples = Y.shape[0]
    I = torch.eye(d).to(Y.device)
    ones = torch.ones(d, d)
    Y_mean = Y.mean(axis=0, keepdims=True)

    reg_val = ortho_lambda * torch.sum(((torch.mm((Y-Y_mean).t(), Y-Y_mean)/(num_samples-1.) - I))**2)

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
                 chunk_cov_estimate=None, max_epochs=2000, dropout=0.5, log_freq=50, batch_size=1024, solver="dnn", device="cpu", dtype=torch.float64, rng_or_seed=None):
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
        self.log_freq = log_freq
        self.batch_size = batch_size
        self.solver = solver
        self.use_rnn = (solver in ['gru', 'bgru', 'lstm', 'blstm'])
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
        #if isinstance(X, list) or X.ndim == 3:
        #    self.mean_ = np.concatenate(X).mean(axis=0, keepdims=True)
        #else:
        #    self.mean_ = X.mean(axis=0, keepdims=True)

        cross_covs = calc_cross_cov_mats_from_data(X, 2 * self.T,
                                                   chunks=self.chunk_cov_estimate,
                                                   regularization=regularization,
                                                   reg_ops=reg_ops)
        delta_time = round((time.time() - start) / 60., 1)
        self._logger.info('Cross covariance estimate took {:0.1f} minutes.'.format(delta_time))

        return cross_covs

    def build_nn_loss(self, Y, ortho_lambda=1., smooth_lambda=1.):
        if Y.ndim == 2: Y = Y.unsqueeze(0)
        losses = torch.zeros(Y.shape[0])
        regs = torch.zeros(Y.shape[0])
        for idx, seq in enumerate(Y):
            c = self.estimate_cross_covariance(seq) # cross covariance: 2T * d * d 
            loss = -calc_pi_from_cross_cov_mats(c) # compute negative pi from the cross covariance
            ortho_reg = ortho_reg_Y(seq, ortho_lambda) # orthognal regularization
            #smooth_reg = smooth_reg_Y(Y, smooth_lambda)
            losses[idx] = loss
            regs[idx] = ortho_reg
        return losses.mean()+regs.mean(), losses.mean()

    def _fit_projection(self, X_train, X_val, Y_val_gt, writer, d=None, record_V=False):
        if d is None:
            d = self.d
        if d < 1:
            raise ValueError

        N = X_train.shape[1]

        if not self.use_rnn:
            model = DNN(N, d, dropout=self.dropout).to(self.device) # Dim reduction NN
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
            X_train = torch.Tensor(X_train).to(self.device)
            X_val = torch.Tensor(X_val).to(self.device)
            batch_size = self.batch_size
            n_batch = int(len(X_train)/batch_size)
            for epoch in range(self.max_epochs):
                tot_pi = 0.
                for i in range(n_batch):
                    print(i, "/", n_batch, X_train[i*batch_size: (i+1)*batch_size].shape)
                    model.train()
                    optimizer.zero_grad()
                    Y_train = model(X_train[i*batch_size: (i+1)*batch_size])
                    loss, neg_pi = self.build_nn_loss(Y_train, self.ortho_lambda, self.smooth_lambda) # build loss and pi for the latent low-dim representations
                    tot_pi += -neg_pi
                    loss.backward()
                    optimizer.step()
                
                if epoch % self.log_freq == 0:
                    print(epoch, ":", -loss.item())
                
                model.eval()
                Y_val = model(X_val)
                self.evaluate(epoch, model, writer, tot_pi, n_batch, Y_val, Y_val_gt, self.ortho_lambda, self.smooth_lambda)
        else:
            model = RNN(idim=N, elayers=2, cdim=128, hdim=d, dropout=self.dropout, typ=self.solver).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            X_train, X_val = torch.Tensor(X_train).to(self.device), torch.Tensor(X_val).to(self.device)
            batch_size = self.batch_size
            X_rnn_train, ls_rnn_train = self.gen_rnn_segs(X_train)
            n_train = len(X_rnn_train)
            n_batch = int(n_train/self.batch_size)
            for epoch in range(self.max_epochs):
                tot_pi = 0.
                for i in range(n_batch):
                    right_end = min((i+1)*batch_size, n_train)
                    #print(i, "/", n_batch, X_rnn_train[i*batch_size:right_end].shape)
                    model.train()
                    optimizer.zero_grad()
                    Y_train, _, _ = model(X_rnn_train[i*batch_size:right_end], ls_rnn_train[i*batch_size:right_end])
                    #print("Y_train:", Y_train.shape)
                    #print("states:", states.shape)
                    loss, neg_pi = self.build_nn_loss(Y_train, self.ortho_lambda, self.smooth_lambda)
                    tot_pi += -neg_pi
                    loss.backward()
                    optimizer.step()
                
                if epoch % self.log_freq == 0:
                    print(epoch, ":", -loss.item())
                
                model.eval()
                X_rnn_val, ls_rnn_val = self.gen_rnn_segs(X_val)
                n_val = len(X_rnn_val)
                n_batch = int(n_val/batch_size)
                Y_val = []
                for i in range(n_batch):
                    seq_val, _, _ = model(X_rnn_val[i*batch_size:(i+1)*batch_size], ls_rnn_val[i*batch_size:(i+1)*batch_size])
                    Y_val.extend(seq_val)
                Y_val = torch.cat(Y_val)
                self.evaluate(epoch, model, writer, tot_pi, n_batch, Y_val, Y_val_gt, self.ortho_lambda, self.smooth_lambda)

        self.model = model
        return Y_val.squeeze()

    def gen_rnn_segs(self, X, n_steps = 1000):
        N = X.shape[1]
        n_len = len(X)
        #X_rnn_train = X.flatten().unfold(0, N*n_steps, N).view(-1, n_steps, N).contiguous()
        #ls_rnn_train = torch.LongTensor([n_steps]*len(X_rnn_train))
        X_rnn_train = torch.zeros(int(n_len/n_steps), n_steps, N).to(X.device)
        ls_rnn_train = torch.zeros(int(n_len/n_steps), dtype=int).to(X.device)
        st_idx = 0
        for i in range(int(n_len/n_steps)):
            X_rnn_train[i] = X[i*n_steps:(i+1)*n_steps]
            ls_rnn_train[i] = n_steps
        #X_rnn_train = torch.FloatTensor(np.array(X_rnn_train)).to(X.device)
        #ls_rnn_train = torch.LongTensor(np.array(ls_rnn_train)).to(X.device)
        #print(X_rnn_train.shape, ls_rnn_train.shape)
        return X_rnn_train, ls_rnn_train

    def evaluate(self, epoch, model, writer, tot_pi, n_batch, Y_val, Y_val_gt, ortho_lambda, smooth_lambda):
        writer.add_scalar('train/train dDCA PI', -tot_pi/n_batch, epoch)
        final_pi = -tot_pi/n_batch

        val_loss, neg_val_pi = self.build_nn_loss(Y_val, ortho_lambda, smooth_lambda) #evaluate loss and pi for the validation set
        writer.add_scalar('val/val dDCA PI', -neg_val_pi, epoch)

        gt_loss, neg_gt_pi = self.build_nn_loss(Y_val_gt, ortho_lambda, smooth_lambda) #evaluate the ground-truth pi
        writer.add_scalar('gt/gt dDCA PI', -neg_gt_pi, epoch)


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
        
        Y_val = self._fit_projection(X_train, X_val, Y_val_gt, writer, d=d) # fit X_train and eval on X_val
        return Y_val

