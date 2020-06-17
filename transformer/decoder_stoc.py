#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""

import pdb
import torch

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import StochasticDecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
# from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.repeat import repeat_i
from espnet.nets.scorer_interface import ScorerInterface

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.e2e_asr import pad_list

class Decoder(ScorerInterface, torch.nn.Module):
    """Transfomer decoder module.

    :param int odim: output dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    :param bool use_output_layer: whether to use output layer
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, odim,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 death_rate=0.0,
                 positional_dropout_rate=0.1,
                 self_attention_dropout_rate=0.0,
                 src_attention_dropout_rate=0.0,
                 input_layer="embed",
                 use_output_layer=True,
                 pos_enc_class=PositionalEncoding,
                 normalize_before=True,
                 concat_after=False):
        """Construct an Decoder object."""
        torch.nn.Module.__init__(self)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        self.decoders = repeat_i(
            num_blocks,
            lambda i: StochasticDecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                death_rate * i / num_blocks,
                normalize_before,
                concat_after
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out) if input_layer == "embed"
                                 input tensor (batch, maxlen_out, #mels) in the other cases
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask,  (batch, maxlen_in)
                                         dtype=torch.uint8 in PyTorch 1.2-
                                         dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :return x: decoded token score before softmax (batch, maxlen_out, token) if use_output_layer is True,
                   final block outputs (batch, maxlen_out, attention_dim) in the other cases
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, cache=None):
        """Forward one step.

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache: cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
            `y.shape` is (batch, maxlen_out, token)
            Weiran: I think the actual size of y is (batch, token), due to y = x[:, -1]
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        x = self.embed(tgt)
        if cache is None:
            cache = self.init_state()
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, None, cache=c)
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    # beam search API (see ScorerInterface)
    def init_state(self, x=None):
        """Get an initial state for decoding."""
        return [None for i in range(len(self.decoders))]

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state)
        return logp.squeeze(0), state

    # Weiran: I created this function for streaming decode.
    def batch_forward_one_step_with_cache(self, ys, caches, memory, memory_mask):
        """Forward one step.

        :param ys: list of list of input token ids, int64
                *Weiran*: I handle tgt_mask myself, to make sure tgt_mask attends correctly.
        :param List[torch.Tensor] cache: cached output list of (batch, max_time_out-1, size)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask,  (batch, maxlen_in)
                                 dtype=torch.uint8 in PyTorch 1.2-
                                 dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :return y, cache: NN output value and cache per `self.decoders`.
                                `y.shape` is (batch, token)
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """

        num_samples = len(ys)
        olens = [len(y) for y in ys]
        tgt = pad_list([torch.tensor(y[::-1]).long() for y in ys], 0).flip(-1)
        # Weiran: the type conversions can be omitted with high version of pytorch.
        tgt_mask = make_non_pad_mask(olens).type(torch.uint8).flip(-1).type(torch.bool).to(memory.device).unsqueeze(-2)
        num_layers = len(self.decoders)
        att_dims = [decoder.size for decoder in self.decoders]

        x = self.embed(tgt)
        if caches is None:
            caches = self.init_state()
        else:
            # First group samples, then group layers.
            caches = [pad_list([ca[i].flip(-2) if ca is not None else torch.zeros([0, att_dims[i]]) for ca in caches], 0.0).
                        flip(-2).to(memory.device) for i in range(num_layers)]

        new_caches = []
        for c, decoder in zip(caches, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask, cache=c)
            new_caches.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        # Weiran: note we are picking the last frames for cache.
        new_caches = [[ca[i, -olens[i]:] for ca in new_caches] for i in range(num_samples)]
        return y, new_caches