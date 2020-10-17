import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dapc.solver import LIN, DNN

def smoothen(raw_xs, window_len=12, window='hamming'):
    xs = raw_xs.T
    ys = []
    for x in xs:
        if window_len < 3:
            return x
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        y = np.convolve(w / w.sum(), s, mode='valid')
        y = y[window_len // 2:-((window_len - 1) // 2)]
        ys.append(y)
    ys = np.array(ys).T
    return ys


def match(X, X_true, max_epochs=3000, device="cpu", verbose=1):  # use a linear mapping to match the reconstructed lorenz attractor and the ground truth attractor
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X).to(device)  # torch tensorize
    if not isinstance(X_true, torch.Tensor):
        X_true = torch.Tensor(X_true).to(device)  # torch tensorize

    match_model = LIN(X.shape[1], X_true.shape[1]).to(device)  # a linear model for matching
    match_opt = torch.optim.Adam(match_model.parameters(), lr=1e-3)  # Adam for optimizing
    for epoch in range(max_epochs):  # one can overfit as much as possible since the optimization is for reconstruction
        X_rec = match_model(X, None)
        match_opt.zero_grad()
        loss = F.mse_loss(X_rec, X_true)  # alternative losses: l1
        loss.backward()
        loss.detach()
        if epoch % 2000 == 0 and verbose == 1:
            print(epoch, ":", loss.item())
        match_opt.step()
    if verbose == 0:
        print("----------------")
        print("final:", loss.item())
        print("----------------")
    return match_model(X, None).detach().cpu().numpy(), loss.item()


def plot_3d_sig(X_dca, plt_idx=300, fig_name="X_dca.png"):  # plot 3-d signals in one figure
    idxs = np.arange(plt_idx)
    plt.plot(idxs, X_dca[:plt_idx, 0], color='b')
    plt.plot(idxs, X_dca[:plt_idx, 1], color='r')
    plt.plot(idxs, X_dca[:plt_idx, 2], color='g')
    plt.savefig("figs/{}".format(fig_name))
    plt.clf()


def split(X, split_rate):
    len_X = len(X)
    return X[:int(len_X * split_rate), :], X[int(len_X * split_rate):, :]


def chunk_long_seq(X, step, sublen=500):
    # X is one long sequence.
    # step is the overlap between chunks.
    # sublen is the length of chunked subsequences.

    total_len, idim = X.shape
    subseqs = []
    sublens = []
    for i in range(0, total_len - sublen, step):
        subseqs.append(X[i:(i + sublen), :])
        sublens.append(sublen)

    return subseqs, sublens


