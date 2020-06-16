import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import logging

# Weiran: moved this function here for now.
def ortho_reg_fn(V, ortho_lambda):
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


class LIN(nn.Module):

    def __init__(self, n_input, n_output, dropout=0.0):
        super(LIN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_input, n_output)

    def forward(self, x, ilens):
        x = self.dropout(x)
        x = self.fc1(x)
        return x, ilens, None


class DNN(nn.Module):
    
    def __init__(self, n_input, n_output, dropout=0.5, n_hid=512):
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_input, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_output)
        #self.lists = nn.ModuleList([self.fc1, self.fc2])

    def forward(self, x, ilens):
        x = self.dropout(F.elu(self.fc1(x)))
        x = self.dropout(F.elu(self.fc2(x)))
        x = self.fc3(x)
        return x, ilens, None

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
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad, ilens, states  # x: utt list of frame x dim

def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states
