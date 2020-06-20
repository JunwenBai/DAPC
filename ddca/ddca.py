import numpy as np
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .solver import LIN, DNN, RNN, ortho_reg_fn
from transformer.encoder_stoc import Encoder
from .cov_utils import calc_cov_from_data, calc_pi_from_cov
from .utils import make_non_pad_mask, pad_list, _context_concat, gen_batch_indices
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
                 encoder_type="dnn", block_toeplitz=True, diag_reg=1e-6, dropout=0.5, use_cpc=False, num_pos=4, num_neg=16,
                 dtype=torch.float32, rng_or_seed=None):
        super(DynamicalComponentsAnalysis, self).__init__()
        self.input_context = input_context
        self.idim = idim * (1+2*input_context)
        self.fdim = fdim
        self.T = T
        self.recon_lambda = recon_lambda
        self.ortho_lambda = ortho_lambda
        self.dropout = dropout
        self.use_cpc = use_cpc
        self.num_pos = num_pos
        self.num_neg = num_neg
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
            if self.encoder_type == "transformer":
                self.encoder = Encoder(
                    idim=idim * (1+2*input_context),
                    attention_dim=256,  # args.adim,
                    attention_heads=4,  # args.aheads,
                    linear_units=2048,  # args.eunits,
                    num_blocks=12,  # args.elayers,
                    input_layer="linear",  # args.transformer_input_layer,
                    dropout_rate=self.dropout,  # args.dropout_rate,
                    death_rate=0.0  # args.edeath_rate
                )
                self.proj = LIN(256, self.fdim, dropout=self.dropout)
            else:
                if self.encoder_type == "dnn":
                    self.encoder = DNN(self.idim, self.fdim, h_sizes=[512, 512], dropout=self.dropout)  # Dim reduction NN
                else:  # ['lstm', 'gru', 'blstm', 'bgru']
                    self.encoder = RNN(idim=self.idim, elayers=3, cdim=256, hdim=self.fdim, dropout=self.dropout,
                                   typ=self.encoder_type)

        if self.use_cpc:
            self.cpc_future_enc = LIN(self.idim, self.fdim, dropout=self.dropout)

        # Weiran: based on my experience, reconstruction network would better be a DNN than RNNs.
        if self.recon_lambda > 0:
            self.decoder = DNN(self.fdim, self.idim, h_sizes=[512, 512], dropout=self.dropout)
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
        if not self.encoder_type == "transformer":
            hs_pad, olens, _ = self.encoder(xs_pad, ilens)
            olens_list = olens.tolist()
            hmask = make_non_pad_mask(olens_list).to(xs_pad.device)
        else:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
            hs_pad, hmask = self.encoder(xs_pad, src_mask)
            hs_pad = self.proj(hs_pad, None)
            hmask = hmask[:, 0, :]
            olens = torch.sum(hmask.long(), 1)

        # Compute cov matrix.
        ortho_loss, self.cov_frame = ortho_reg_Y(hs_pad, hmask)
        if self.encoder_type == "lin":
            ortho_loss = ortho_reg_fn(self.encoder.fc1.weight.t())

        if self.decoder:
            recon, _, _ = self.decoder(hs_pad, olens)
            recon_loss = compute_recon_mse(recon, xs_pad, hmask)
        else:
            recon_loss = 0.0
        
        #print("hs_pad:", hs_pad.shape)
        if not self.use_cpc:
            self.cov = calc_cov_from_data(hs_pad, hmask, 2 * self.T, toeplitzify=self.block_toeplitz, reg=self.diag_reg)
            pi = calc_pi_from_cov(self.cov)
            key_loss = - pi
        else:
            pi = 0.
            slfidx, posidx, negidx = gen_batch_indices(ilens, max(ilens), range(self.num_pos), self.num_neg, portion=1.0)
            fx = hs_pad.view(-1, hs_pad.shape[-1])
            gy = self.cpc_future_enc(xs_pad.view(-1, xs_pad.shape[-1]))
            fx_slf = fx[slfidx]
            gy_pos = gy[posidx]
            pos_score = torch.sum(fx_slf * gy_pos, 1, keepdim=True)
            fx_slf = fx_slf.repeat(1, self.num_neg).view(-1, self.fdim)
            gy_neg = gy[negidx]
            neg_score = torch.sum(fx_slf * gy_neg, 1).view(-1, self.num_neg)
            scores = torch.cat([pos_score, neg_score], 1)
            log_preds = F.log_softmax(scores)
            key_loss = - torch.mean(log_preds[:, 0])

        self.loss = key_loss + self.ortho_lambda * ortho_loss + self.recon_lambda * recon_loss
        return self.loss, float(pi), float(ortho_loss), float(recon_loss), (self.cov_frame).detach().cpu().numpy()


    def encode(self, x):
        # Weiran: encode only one utterance.
        # x is a 2D tensor of shape time x idim.
        self.eval()
        ilens = torch.tensor([x.size(0)], device=x.device).long()
        if not self.encoder_type == "transformer":
            hs_pad, _, _ = self.encoder(x.unsqueeze(0), ilens)
        else:
            enc_output, _ = self.encoder(x.unsqueeze(0), None)
            hs_pad = self.proj(enc_output, None)
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
        total_loss = 0.0
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

            total_loss += loss
            total_pi += pi
            total_ortho_loss += loss_orth
            total_loss_recon += loss_recon * total_len_batch

        avg_loss_train = total_loss / n_batch_train
        avg_pi_train = total_pi / n_batch_train
        avg_ortho_loss_train = total_ortho_loss / n_batch_train
        avg_recon_loss_train = total_loss_recon / sum(L_train)
        print("epoch %d, train avg loss=%f, pi=%f, ortho_loss=%f, recon_loss=%f" %
              (epoch, avg_loss_train, avg_pi_train, avg_ortho_loss_train, avg_recon_loss_train))

        model.eval()
        total_loss = 0.0
        total_pi = 0.0
        total_ortho_loss = 0.0
        total_loss_recon = 0.0
        total_cov_frame = np.zeros([model.fdim, model.fdim])
        for i in range(int(math.ceil(n_valid / batch_size))):
            x_batch = [torch.from_numpy(_context_concat(X_valid[_],input_context)).float() for _ in range(i*batch_size, min((i+1)*batch_size, n_valid))]
            l_batch = [L_valid[_] for _ in range(i*batch_size, min((i+1)*batch_size, n_valid))]
            total_len_batch = sum(l_batch)
            x_batch, l_batch = pad_list(x_batch, 0.0).to(device), torch.Tensor(l_batch).long().to(device)

            loss, pi, loss_orth, loss_recon, cov_frame = model(x_batch, l_batch)
            #print("minibatch %03d/%03d: pi=%f" % (i, n_batch_valid, pi))

            total_loss += loss.detach()
            total_pi += pi
            total_ortho_loss += loss_orth
            total_loss_recon += loss_recon * total_len_batch
            total_cov_frame += cov_frame * total_len_batch

        avg_loss_valid = total_loss / n_batch_valid
        avg_pi_valid = total_pi / n_batch_valid
        avg_ortho_loss_valid = total_ortho_loss / n_batch_valid
        avg_recon_loss_valid = total_loss_recon / sum(L_valid)
        avg_cov_frame = total_cov_frame / sum(L_valid)
        print("epoch %d, valid avg loss=%f, pi=%f, ortho_loss=%f, recon_loss=%f" %
              (epoch, avg_loss_valid, avg_pi_valid, avg_ortho_loss_valid, avg_recon_loss_valid))
        print(avg_cov_frame)

        # Write stats.
        writer.add_scalar('train/pi', avg_pi_train, epoch)
        writer.add_scalar('train/orth', avg_ortho_loss_train, epoch)
        writer.add_scalar('train/recon', avg_recon_loss_train, epoch)
        writer.add_scalar('valid/pi', avg_pi_valid, epoch)
        writer.add_scalar('valid/orth', avg_ortho_loss_valid, epoch)
        writer.add_scalar('valid/recon', avg_recon_loss_valid, epoch)

    return model
