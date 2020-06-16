import time
import numpy as np
import math
import logging

import torch

from .solver import LIN, DNN, RNN
from .cov_utils import calc_cov_from_data, calc_pi_from_cov
from .utils import make_non_pad_mask, pad_list, _context_concat
import pdb


def ortho_reg_Y(Y, src_mask):
    # Weiran: Y is of shape (B, maxlen, fdim).
    fdim = Y.size(2)
    I = torch.eye(fdim, device=Y.device, dtype=Y.dtype)

    Y = torch.reshape(Y, [-1, fdim])
    mask_float = src_mask.float().view([-1, 1])
    Y_mean = torch.sum(torch.mul(Y, mask_float), 0, keepdim=True) / torch.sum(mask_float)
    Y = Y - Y_mean

    cov = torch.mm(torch.mul(Y, mask_float).t(), Y) / torch.sum(mask_float)
    return torch.sum((cov - I) ** 2), cov


def compute_recon_mse(recon, X, src_mask):
    # Weiran: recon and X are of shape (B, maxlen, fdim).
    idim = X.size(2)

    loss = torch.sum( (recon.view([-1,idim]) - X.view([-1,idim])) ** 2, 1)
    mask_float = src_mask.float().view([-1])
    return torch.sum(torch.mul(loss, mask_float)) / torch.sum(mask_float)


class DynamicalComponentsAnalysis(torch.nn.Module):
    """Dynamical Components Analysis.

    Runs DCA on multidimensional time series data X to discover a projection
    onto a fdim-dimensional subspace which maximizes the complexity, as defined by the Gaussian
    Predictive Information (PI) of the fdim-dimensional dynamics over windows of length T.

    Parameters
    ----------
    idim : int
        Sequence's input dimensionality.
    fdim : int
        Number of basis vectors onto which the data X are projected.
    T : int
        Size of time windows across which to compute mutual information. Total window length will be
        `2 * T`. When fitting a model, the length of the shortest time series must be greater than
        `2 * T` and for good performance should be much greater than `2 * T`.
    ortho_lambda : float
        Coefficient on term that keeps V close to orthonormal.
    block_toeplitz : bool
        If True, uses the block-Toeplitz logdet algorithm which is typically faster and less
        memory intensive on cpu for `T >~ 10` and `fdim >~ 40`.
    dtype : pytorch.dtype
        What dtype to use for computation.
    """

    def __init__(self, idim, fdim, T, input_context=0, recon_lambda=0.0, ortho_lambda=0.0,
                 encoder_type="dnn", block_toeplitz=True, diag_reg=1e-6, dropout=0.5,
                 dtype=torch.float32, rng_or_seed=None):
        super(DynamicalComponentsAnalysis, self).__init__()
        self.input_context = input_context
        self.idim = idim * (1+2*input_context)
        self.fdim = fdim
        self.T = T
        self.recon_lambda = recon_lambda
        self.ortho_lambda = ortho_lambda
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.dtype = dtype
        self.block_toeplitz = block_toeplitz
        self.diag_reg = diag_reg
        self.cross_covs = None

        if rng_or_seed is None:
            self.rng = np.random
        elif isinstance(rng_or_seed, np.random.RandomState):
            self.rng = rng_or_seed
        else:
            self.rng = np.random.RandomState(rng_or_seed)

        if self.encoder_type == "lin":
            self.encoder = LIN(self.idim, self.fdim, dropout=self.dropout)
        else:
            if self.encoder_type == "dnn":
                self.encoder = DNN(self.idim, self.fdim, h_sizes=[256, 256], dropout=self.dropout)  # Dim reduction NN
            else:  # ['lstm', 'gru', 'blstm', 'bgru']
                self.encoder = RNN(idim=self.idim, elayers=3, cdim=128, hdim=self.fdim, dropout=self.dropout,
                               typ=self.encoder_type)

        # Weiran: based on my experience, reconstruction network would better be a DNN than RNNs.
        if self.recon_lambda > 0:
            self.decoder = DNN(self.fdim, self.idim, h_sizes=[256, 256], dropout=self.dropout)
        else:
            self.decoder = None

    def forward(self, xs_pad, ilens):
        """ forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, maxlen, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :return: total loss value
        :rtype: torch.Tensor
        :return: PI loss value
        :rtype: float
        :return: regularization value
        :rtype: float
        """

        # Weiran: for now, both DNN and RNNs do not reduce the lengths.
        hs_pad, olens, _ = self.encoder(xs_pad, ilens)
        olens_list = olens.tolist()
        hmask = make_non_pad_mask(olens_list).to(xs_pad.device)

        # Compute cov matrix.
        self.cov = calc_cov_from_data(hs_pad, hmask, 2 * self.T, toeplitzify=self.block_toeplitz, reg=self.diag_reg)

        pi = calc_pi_from_cov(self.cov)
        ortho_loss, self.cov_frame = ortho_reg_Y(hs_pad, hmask)

        if self.decoder:
            recon, _, _ = self.decoder(hs_pad, olens)
            recon_loss = compute_recon_mse(recon, xs_pad, hmask)
        else:
            recon_loss = 0.0
        self.loss = - pi + self.ortho_lambda * ortho_loss + self.recon_lambda * recon_loss
        return self.loss, float(pi), float(ortho_loss), float(recon_loss), (self.cov_frame).detach().cpu().numpy()


    def encode(self, x):
        # Weiran: encode only one utterance.
        # x is a 2D tensor of shape time x idim.
        self.eval()
        ilens = torch.tensor([x.size(0)], device=x.device).long()
        hs_pad, olens, _ = self.encoder(x.unsqueeze(0), ilens)
        return hs_pad.squeeze(0).detach()


# Move training code out of model definition.
def fit_ddca(model, X_train, L_train, X_valid, L_valid, writer, use_gpu=False,
             batch_size=50, max_epochs=500):
    if use_gpu:
        model = model.cuda()
    device = torch.device("cuda" if use_gpu else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    input_context = model.input_context

    # X_train is a sequence of input sequences, whose lengths are saved in L_train.
    # The proper way is to write a data loader to get next batch.
    n_train = len(X_train)
    n_batch_train = int(math.ceil(n_train / batch_size))
    n_valid = len(X_valid)
    n_batch_valid = int(math.ceil(n_valid / batch_size))

    for epoch in range(max_epochs):
        model.train()
        order = np.random.permutation(n_train)
        total_pi = 0.0
        total_ortho_loss = 0.0
        total_loss_recon = 0.0
        
        for i in range(n_batch_train):
            idx_batch = list(order[i * batch_size: min((i + 1) * batch_size, n_train)])

            x_batch = [torch.from_numpy(_context_concat(X_train[_], input_context)).float() for _ in idx_batch]
            l_batch = [L_train[_] for _ in idx_batch]
            total_len_batch = sum(l_batch)
            x_batch, l_batch = pad_list(x_batch, 0.0).to(device), torch.Tensor(l_batch).long().to(device)

            optimizer.zero_grad()
            loss, pi, loss_orth, loss_recon, _ = model(x_batch, l_batch)
            #print("minibatch %03d/%03d: pi=%f" % (i, n_batch_train, pi))
            loss.backward()
            loss.detach()
            optimizer.step()

            total_pi += pi
            total_ortho_loss += loss_orth
            total_loss_recon += loss_recon * total_len_batch

        avg_pi_train = total_pi / n_batch_train
        avg_ortho_loss_train = total_ortho_loss / n_batch_train
        avg_recon_loss_train = total_loss_recon / sum(L_train)
        print("epoch %d, train avg pi=%f, ortho_loss=%f, recon_loss=%f" %
              (epoch, avg_pi_train, avg_ortho_loss_train, avg_recon_loss_train))

        model.eval()
        total_pi = 0.0
        total_ortho_loss = 0.0
        total_loss_recon = 0.0
        total_cov_frame = np.zeros([model.fdim, model.fdim])
        for i in range(int(math.ceil(n_valid / batch_size))):
            x_batch = [torch.from_numpy(_context_concat(X_valid[_],input_context)).float() for _ in range(i*batch_size, min((i+1)*batch_size, n_valid))]
            l_batch = [L_valid[_] for _ in range(i*batch_size, min((i+1)*batch_size, n_valid))]
            total_len_batch = sum(l_batch)
            x_batch, l_batch = pad_list(x_batch, 0.0).to(device), torch.Tensor(l_batch).long().to(device)

            _, pi, loss_orth, loss_recon, cov_frame = model(x_batch, l_batch)
            #print("minibatch %03d/%03d: pi=%f" % (i, n_batch_valid, pi))

            total_pi += pi
            total_ortho_loss += loss_orth
            total_loss_recon += loss_recon * total_len_batch
            total_cov_frame += cov_frame * total_len_batch

        avg_pi_valid = total_pi / n_batch_valid
        avg_ortho_loss_valid = total_ortho_loss / n_batch_valid
        avg_recon_loss_valid = total_loss_recon / sum(L_valid)
        avg_cov_frame = total_cov_frame / sum(L_valid)
        print("epoch %d, valid avg pi=%f, ortho_loss=%f, recon_loss=%f" %
              (epoch, avg_pi_valid, avg_ortho_loss_valid, avg_recon_loss_valid))
        print(avg_cov_frame)

        # Write stats.
        writer.add_scalar('train/pi', avg_pi_train, epoch)
        writer.add_scalar('train/orth', avg_ortho_loss_train, epoch)
        writer.add_scalar('train/recon', avg_recon_loss_train, epoch)
        writer.add_scalar('valid/pi', avg_pi_valid, epoch)
        writer.add_scalar('valid/orth', avg_ortho_loss_valid, epoch)
        writer.add_scalar('valid/recon', avg_recon_loss_valid, epoch)

    return model
