#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Weiran Wang.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Optimizer module."""

import torch
import math


class RampupOpt(object):
    """Optim wrapper that implements long training schedule in Sec 3.2 of the SpecAugment paper:
        https://arxiv.org/pdf/1904.08779.pdf."""

    def __init__(self, model_size, sr, si, sf, lr, optimizer):
        """Construct an NoamOpt object."""
        self.optimizer = optimizer
        self._step = 0
        self.sr = sr
        self.si = si
        self.sf = sf
        # Peak learning rate.
        self.lr = lr
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step

        if step <= self.sr:  # rampup phase
            return (self.lr / self.sr) * step
        elif step <= self.si:
            return self.lr
        elif step <= self.sf:
            return self.lr * math.exp(math.log(0.01) / (self.sf - self.si) * (step - self.si))
        else:
            return self.lr * 0.01

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "sr": self.sr,
            "si": self.si,
            "sf": self.sf,
            "lr": self.lr,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_std_opt(model, d_model, sr, si, sf, lr):
    """Get standard NoamOpt."""
    base = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return RampupOpt(d_model, sr, si, sf, lr, base)
