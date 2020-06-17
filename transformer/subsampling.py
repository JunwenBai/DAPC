#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""

import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding

import numpy as np
def _context_concat(seq, context_size=0):
    """ seq is of size length x feat_dim.
    output is of size length x (feat_dim*(1+2*context_size)).
    """

    if context_size == 0:
        return seq

    output = []
    length = seq.shape[0]
    # Left concatenation.
    for j in range(context_size):
        tmp = np.concatenate([np.repeat(seq[np.newaxis, 0, :], j + 1, axis=0), seq[0:(length - j - 1), :]], 0)
        output.append(tmp)

    # Add original inputs.
    output.append(seq)

    # Right concatenation.
    for j in range(context_size):
        tmp = np.concatenate([seq[(j + 1):length, :],
                              np.repeat(seq[np.newaxis, length - 1, :], j + 1, axis=0)], 0)
        output.append(tmp)

    return np.concatenate(output, 1)


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        if x_mask.size(1)==1:
            return x, x_mask[:, :, :-2:2][:, :, :-2:2]
        else:
            # Weiran: if the mask is full, both time dimensions need to be subsampled.
            return x, x_mask[:, :-2:2, :-2:2][:, :-2:2, :-2:2]


# Weiran: I wrote this module to perform subsampling for delta features.
class Conv2dSubsampling_with_deltas(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling_with_deltas, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim//3 - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        b, t, f = x.size()
        x = x.view(b, t, 3, f // 3).transpose(1, 2)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


# Weiran: I created this module to have less subsampling than the one above.
class Conv2dSubsampling_1layer_with_deltas(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate, delta=True):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling_1layer_with_deltas, self).__init__()
        self.delta = delta
        if delta:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, odim, 3, 2),
                torch.nn.ReLU()
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(odim * ((idim//3 - 1) // 2), odim),
                PositionalEncoding(odim, dropout_rate)
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2),
                torch.nn.ReLU()
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(odim * ((idim - 1) // 2), odim),
                PositionalEncoding(odim, dropout_rate)
            )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """

        if self.delta:
            b, t, f = x.size()
            x = x.view(b, t, 3, f//3).transpose(1, 2)  # (b, c, t, f)
        else:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2]


# Weiran: Yingbo's code below.
class VariationalDropout2D(torch.nn.Module):
    '''
    assumes the last dimension is time,
    input in shape B x C x H x T
    '''

    def __init__(self, p_drop, seed=None):
        super(VariationalDropout2D, self).__init__()
        self.p_drop = p_drop
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    def forward(self, x):
        if self.training:
            mask = x.data.new(x.size(0), 1, x.size(2), 1).bernoulli_(1 - self.p_drop)
            return x * mask
        else:
            return x * (1 - self.p_drop)

    def init_params(self):
        pass

    def __repr__(self):
        repstr = self.__class__.__name__ + ' (\n'
        repstr += "{:.2f}".format(self.p_drop)
        repstr += ')'
        return repstr


class ResCNNBlock2D(torch.nn.Module):

    def __init__(self, in_feat, out_feat, kernel_size, stride, dropout=0, seed=None):
        super(ResCNNBlock2D, self).__init__()
        self.kernel_size = kernel_size
        f_h, f_w = kernel_size
        self.drop = None
        if dropout > 0:
            self.drop = VariationalDropout2D(dropout, seed)

        layers = [
            torch.nn.BatchNorm2d(in_feat),
            # only pad the time in front, which means left
            # kernel size is in shape of (h, w)
            # where as the padding is padding layer is w and h
            # so a bit weird format
            torch.nn.ConstantPad2d((f_w - 1, 0, f_h // 2, f_h // 2), 0),
            torch.nn.Conv2d(in_feat,
                in_feat,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_feat,
                bias=False),

            torch.nn.BatchNorm2d(in_feat),
            torch.nn.Conv2d(in_feat,
                out_feat,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),

            torch.nn.BatchNorm2d(out_feat),
            torch.nn.ReLU()]
        self.sep_conv = torch.nn.Sequential(*layers)

        s_h, s_w = stride
        self.res_conv = None
        if s_h!=1 or s_w!=1 or in_feat!=out_feat:
            res_layers = [
                torch.nn.BatchNorm2d(in_feat),
                torch.nn.Conv2d(in_feat,
                    out_feat,
                    kernel_size=1,
                    stride=stride)
            ]
            self.res_conv = torch.nn.Sequential(*res_layers)

        self.init_params()

    def forward(self, x):
        fx = self.sep_conv(x)
        if self.res_conv is not None:
            res_x = self.res_conv(x)

        else:
            res_x = x

        # res_x = x if self.res_conv is None else self.res_conv(x)
        ret_x = fx + res_x if self.drop is None else self.drop(fx + res_x)
        return ret_x

    def init_params(self):
        torch.nn.init.kaiming_uniform_(self.sep_conv[2].weight)
        torch.nn.init.kaiming_uniform_(self.sep_conv[4].weight)
        if self.res_conv is not None:
            torch.nn.init.kaiming_uniform_(self.res_conv[1].weight)


class Conv2dSubsampling_yingbo(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling_yingbo, self).__init__()
        self.conv = torch.nn.Sequential(
            ResCNNBlock2D(1, 64, (3, 3), (1, 1), dropout=dropout_rate),
            torch.nn.MaxPool2d(2),
            ResCNNBlock2D(64, 128, (3, 3), (1, 1), dropout=dropout_rate),
            torch.nn.MaxPool2d(2)
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(128 * (idim // 2 // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, 1::2][:, :, 1::2]
