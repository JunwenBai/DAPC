#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch

from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate,
                 normalize_before=True, concat_after=False):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x, mask):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask


class StochasticEncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate, death_rate=0.0,
                 normalize_before=True, concat_after=False):
        """Construct an EncoderLayer object."""
        super(StochasticEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.death_rate = death_rate
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x, mask):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask)), dim=-1)
            if not coin:
                x = residual + self.concat_linear(x_concat) * 0.0
            else:
                if self.training:
                    x = residual + self.concat_linear(x_concat) / (1 - self.death_rate)
                else:
                    x = residual + self.concat_linear(x_concat)
        else:
            if not coin:
                x = residual + self.dropout(self.self_attn(x, x, x, mask)) * 0.0
            else:
                if self.training:
                    x = residual + self.dropout(self.self_attn(x, x, x, mask)) / (1 - self.death_rate)
                else:
                    x = residual + self.dropout(self.self_attn(x, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if not coin:
            x = residual + self.dropout(self.feed_forward(x)) * 0.0
        else:
            if self.training:
                x = residual + self.dropout(self.feed_forward(x)) / (1 - self.death_rate)
            else:
                x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask
