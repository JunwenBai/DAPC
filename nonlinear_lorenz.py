import sys
import pdb

sys.path.append(".")
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from ddca.ddca import DynamicalComponentsAnalysis
from ddca.ddca import fit_ddca
from ddca.utils import _context_concat
from ddca.data_gen import gen_nonlinear_noisy_lorenz, gen_lorenz_data
from models.DNN import DNN, Match_DNN
from plotter import plot_figs
from dca import DynamicalComponentsAnalysis as DCA

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--fdim", default=3, help="Dimensionality of features", type=int)
parser.add_argument("--T", default=4, help="Time steps for estimating PI", type=int)
parser.add_argument("--ortho_lambda", default=10.0, help="Regularization parameter for orthogonality", type=float)
parser.add_argument("--recon_lambda", default=10.0, help="Regularization parameter for reconstruction", type=float)
parser.add_argument("--dropout", default=0.0, help="Dropout probability of networks.", type=float)
parser.add_argument("--batchsize", default=20, help="Number of sequences in each minibatch for unsupervised loss", type=int)
parser.add_argument("--encoder_type", default="lin", type=str, choices=["lin", "dnn", "gru", "lstm", "bgru", "blstm"])
parser.add_argument("--epochs", default=20, help="Number of training epochs", type=int)
parser.add_argument("--input_context", default=0, help="Number of context frames for splicing", type=int)
parser.add_argument("--seed", default=0, help="Random seed", type=int)
args = parser.parse_args()

import torch.nn as nn
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


def match(X, X_true, max_epochs=3000, device="cpu"):  # use a linear mapping to match the reconstructed lorenz attractor and the ground truth attractor
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X).to(device)  # torch tensorize
    if not isinstance(X_true, torch.Tensor):
        X_true = torch.Tensor(X_true).to(device)  # torch tensorize

    match_model = Match_DNN(X.shape[1], X_true.shape[1]).to(device)  # a linear model for matching
    match_opt = torch.optim.Adam(match_model.parameters(), lr=1e-3)  # Adam for optimizing
    for epoch in range(max_epochs):  # one can overfit as much as possible since the optimization is for reconstruction
        X_rec = match_model(X)
        match_opt.zero_grad()
        loss = F.mse_loss(X_rec, X_true)  # alternative losses: l1
        loss.backward()
        loss.detach()
        if epoch % 2000 == 0:
            print(epoch, ":", loss.item())
        match_opt.step()

    return match_model(X).detach().cpu().numpy()


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


if __name__ == "__main__":

    np.random.seed(args.seed)  # fix the seed
    torch.manual_seed(args.seed)  # fix the seed
    torch.cuda.manual_seed(args.seed)

    T = args.T
    fdim = args.fdim
    dropout = args.dropout

    idim = 30  # lift projection dim
    noise_dim = 7  # noisify raw DCA
    split_rate = 0.8
    snr_vals = [10.]  # signal-to-noise ratios
    num_samples = 10000  # samples to collect from the lorenz system

    print("Generating ground truth dynamics ...")
    X_dynamics = gen_lorenz_data(num_samples)  # 10000 * 3
    # noisy_model = DNN(X_dynamics.shape[1], idim)  # DNN lift projection: 3 -> 30 for d-DCA
    # pdb.set_trace()
    noisy_model = KERNEL(X_dynamics[::50], np.arange(0.05, 0.34, 0.01))

    dca_recons = []
    ddca_recons = []
    r2_vals = np.zeros((len(snr_vals), 2))  # obtain R2 scores for DCA and dDCA
    for snr_idx, snr in enumerate(snr_vals):
        print("Generating noisy data with snr=%.2f ..." % snr)
        X_clean, X_noisy = gen_nonlinear_noisy_lorenz(idim, T, snr, X_dynamics=X_dynamics, noisy_model=noisy_model,
                                                      seed=args.seed)
        X_noisy = X_noisy - X_noisy.mean(axis=0)

        X_clean_train, X_clean_val = split(X_clean, split_rate)
        X_noisy_train, X_noisy_val = split(X_noisy, split_rate)
        X_dyn_train, X_dyn_val = split(X_dynamics, split_rate)
        writer = SummaryWriter('runs/ddca_T=%d_reg=%.2f' % (T, args.ortho_lambda))

        # deep DCA
        use_gpu = True
        if use_gpu:
            device = torch.device("cuda:0")
        print("Training d-DCA")
        ddca_model = DynamicalComponentsAnalysis(idim, fdim=fdim, T=T, encoder_type=args.encoder_type,
                                                 input_context=args.input_context,
                                                 ortho_lambda=args.ortho_lambda, recon_lambda=args.recon_lambda,
                                                 dropout=args.dropout, block_toeplitz=False)

        # Weiran: chunk long sequences to shorter ones.
        chunk_size = 500
        X_train_seqs, L_train = chunk_long_seq(X_noisy_train, 30, chunk_size)
        X_valid_seqs, L_valid = chunk_long_seq(X_noisy_val, 30, chunk_size)
        X_clean_seqs, L_clean = chunk_long_seq(X_clean_val, 30, chunk_size)
        X_dyn_seqs, L_dyn = chunk_long_seq(X_dyn_val, 30, chunk_size)

        ddca_model = fit_ddca(ddca_model, X_train_seqs, L_train, X_valid_seqs[:1], L_valid[:1], writer, use_gpu,
                              batch_size=args.batchsize, max_epochs=args.epochs)

        X_ddca = ddca_model.encode(
            torch.from_numpy(_context_concat(X_valid_seqs[0], args.input_context)).float().to(device, dtype=ddca_model.dtype)).cpu()
        print(X_ddca)
        print(torch.mm((X_ddca - X_ddca.mean(0, keepdim=True)).t(), (X_ddca - X_ddca.mean(0, keepdim=True))) / X_ddca.size(0))
        # X_ddca = smoothen(X_ddca)

        # Linear DCA
        print("Training DCA")
        """
        opt = DCA(T=T, d=3, use_scipy=False, block_toeplitz=False, ortho_lambda=10., init="random_ortho",
                  max_epochs=1000, device=device)
        opt.fit(X_noisy_train, X_noisy_val, X_dyn_val, writer)
        V_dca = opt.coef_  # transformation matrix
        X_dca = np.dot(X_noisy_val, V_dca)  # recontructed 3-d signals: X_dca
        X_dca = X_dca[:chunk_size, :]
        # X_dca = smoothen(X_dca)
        """

        dca_model = DynamicalComponentsAnalysis(idim, fdim=fdim, T=T, encoder_type="lin",
                                                 input_context=args.input_context,
                                                 ortho_lambda=10.0, block_toeplitz=False,
                                                 dropout=0.0)
        dca_model = fit_ddca(dca_model, X_train_seqs, L_train, X_valid_seqs[:1], L_valid[:1], writer, use_gpu,
                              batch_size=args.batchsize, max_epochs=args.epochs)

        X_dca = dca_model.encode(
            torch.from_numpy(_context_concat(X_valid_seqs[0], args.input_context)).float().to(device,
                                                                            dtype=dca_model.dtype)).cpu()


        print("Matching DCA")
        X_dca_recon = match(X_dca.detach().cpu().numpy(), X_dyn_seqs[0], 15000, device)
        # match d-DCA with ground-truth
        print("Matching d-DCA")
        X_ddca_recon = match(X_ddca.detach().cpu().numpy(), X_dyn_seqs[0], 15000, device)

        # R2 of dca
        r2_dca = 1 - np.sum((X_dca_recon - X_dyn_seqs[0]) ** 2) / np.sum(
            (X_dyn_seqs[0] - np.mean(X_dyn_seqs[0], axis=0)) ** 2)
        # R2 of ddca
        r2_ddca = 1 - np.sum((X_ddca_recon - X_dyn_seqs[0]) ** 2) / np.sum(
            (X_dyn_seqs[0] - np.mean(X_dyn_seqs[0], axis=0)) ** 2)
        # store R2's
        r2_vals[snr_idx] = [r2_dca, r2_ddca]
        # store reconstructed signals    
        dca_recons.append(X_dca_recon)
        ddca_recons.append(X_ddca_recon)

    plot_figs(dca_recons, ddca_recons, X_dyn_seqs[0], X_clean_seqs[0], X_valid_seqs[0], r2_vals, snr_vals, "DCA",
              "d-DCA", "figs/result.pdf")

"""
python3 nonlinear_lorenz.py --encoder_type dnn --dropout 0.5 --ortho_lambda 100.0
"""