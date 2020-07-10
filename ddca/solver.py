import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from .utils import make_non_pad_mask
from transformer.encoder_stoc import Encoder

import math
import logging


def ortho_reg_Y(Y, src_mask):
    # Weiran: Y is of shape (B, maxlen, fdim).
    fdim = Y.size(2)
    I = torch.eye(fdim, device=Y.device, dtype=Y.dtype)
    ones = torch.ones_like(I)

    Y = torch.reshape(Y, [-1, fdim])
    mask_float = src_mask.float().view([-1, 1])
    Y_mean = torch.sum(torch.mul(Y, mask_float), 0, keepdim=True) / torch.sum(mask_float)
    Y = Y - Y_mean

    cov = torch.mm(torch.mul(Y, mask_float).t(), Y) / torch.sum(mask_float)
    return torch.sum((cov - I) ** 2), cov
    # return torch.sum(((cov - I) ** 2) * (ones - I)), cov


def ortho_reg_fn(V, ortho_lambda=1.):
    """Regularization term which encourages the basis vectors in the
    columns of V to be orthonormal.
    Parameters
    ----------
    V : shape (hidden, fdim)
        Projection layer.
    ortho_lambda : float
        Regularization hyperparameter.
    Returns
    -------
    reg_val : float
        Value of regularization function.
    """

    fdim = V.shape[1]
    reg_val = ortho_lambda * torch.sum((torch.mm(V.t(), V) - torch.eye(fdim, device=V.device, dtype=V.dtype)) ** 2)

    return reg_val


class KERNEL(nn.Module):

    def __init__(self, centers, sigmas):
        super(KERNEL, self).__init__()
        self.C = centers.shape[0]
        self.S = len(sigmas)
        self.centers = centers
        self.sigmas = sigmas
        self.list = nn.ModuleList([torch.nn.Linear(self.C, 1) for _ in range(self.S)])
        self.reset_parameters()

    def reset_parameters(self, stdv=1.0):
        for layer in self.list:
            layer.weight.data.normal_(stdv)
            if layer.bias is not None:
                layer.bias.data.normal_(stdv)

    def forward(self, x):
        diff = x.unsqueeze(1) - torch.tensor(self.centers, dtype=x.dtype).unsqueeze(0)
        # N x C matrix.
        sqdist = torch.sum(diff ** 2, 2)
        # N x S x C tensor.
        aff = torch.exp(- sqdist.unsqueeze(1) / torch.reshape(torch.tensor(self.sigmas, dtype=sqdist.dtype), [1, -1, 1]))
        return torch.cat([self.list[i](aff[:,i,:]) for i in range(self.S)], 1)


class LIN(nn.Module):

    def __init__(self, n_input, n_output, dropout=0.0):
        super(LIN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_input, n_output)

    def forward(self, x, ilens=None):
        x = self.dropout(x)
        x = self.fc1(x)
        if ilens is None:
            return x
        else:
            return x, ilens


class DNN(nn.Module):
    
    def __init__(self, n_input, n_output, dropout=0.0, h_sizes=[128, 128], reset_param=False):
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(n_input, h_sizes[0])
        self.hidden = nn.ModuleList([self.fc_in])
        self.n_layers = len(h_sizes)
        for i in range(1, self.n_layers):
            self.hidden.append(nn.Linear(h_sizes[i-1], h_sizes[i]))
        self.fc_out = nn.Linear(h_sizes[-1], n_output)
        if reset_param:
            self.reset_parameters()
        
    def reset_parameters(self, std=0.2):
        for layer in self.hidden:
            layer.weight.data.normal_(std)
            if layer.bias is not None:
                layer.bias.data.normal_(std)

    def forward(self, x, ilens=None):
        for fc in self.hidden:
            x = self.dropout(F.elu(fc(x)))
        x = self.fc_out(x)
        if ilens is None:
            return x
        else:
            return x, ilens


class RNN(torch.nn.Module):
    """RNN module
    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                   dropout=dropout, bidirectional=bidir) if "lstm" in typ \
            else torch.nn.GRU(idim, cdim, elayers, batch_first=True, dropout=dropout,
                              bidirectional=bidir)
        if bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim, hdim)
        #torch.nn.init.constant_(self.l_last.bias.data, -5.)
        self.typ = typ

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNN forward
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed, it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, states = self.nbrnn(xs_pack, hx=prev_state)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        #projected = torch.tanh(self.l_last(ys_pad.contiguous().view(-1, ys_pad.size(2))))*4.
        projected = self.l_last(ys_pad.contiguous().view(-1, ys_pad.size(2)))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad, ilens
        # Weiran: not returning states.
        # return xs_pad, ilens, states  # x: utt list of frame x dim


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states


class TRANSFORMER(nn.Module):

    def __init__(self, idim, odim, adim, aheads, eunits, elayers, input_layer, dropout_rate, death_rate=0.0):
        super(TRANSFORMER, self).__init__()
        self.encoder = Encoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=input_layer,
            dropout_rate=dropout_rate,
            death_rate=death_rate,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(adim, odim)

    def forward(self, xs_pad, ilens):
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hmask = self.encoder(xs_pad, src_mask)
        hs_pad = self.proj(self.dropout(hs_pad))
        hmask = hmask[:, 0, :]
        olens = torch.sum(hmask, 1)

        return hs_pad, olens
