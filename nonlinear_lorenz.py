import sys
import pdb

sys.path.append(".")
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from ddca.ddca import DynamicalComponentsAnalysis
from ddca.ddca import fit_ddca
from ddca.data_gen import gen_nonlinear_noisy_lorenz, gen_lorenz_data
from models.DNN import DNN, Match_DNN
from plotter import plot_figs
from dca import DynamicalComponentsAnalysis as DCA

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


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


def match(X, X_true,
          max_epochs=3000):  # use a linear mapping to match the reconstructed lorenz attractor and the ground truth attractor
    if not isinstance(X, torch.Tensor): X = torch.Tensor(X)  # torch tensorize
    if not isinstance(X_true, torch.Tensor): X_true = torch.Tensor(X_true)  # torch tensorize

    match_model = Match_DNN(X.shape[1], X_true.shape[1])  # a linear model for matching
    match_opt = torch.optim.Adam(match_model.parameters(), lr=1e-3)  # Adam for optimizing
    for epoch in range(max_epochs):  # one can overfit as much as possible since the optimization is for reconstruction
        X_rec = match_model(X)
        match_opt.zero_grad()
        loss = F.mse_loss(X_rec, X_true)  # alternative losses: l1
        loss.backward()
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
    # Set parameters
    T = int(sys.argv[1])  # time window
    ortho_lambda = float(sys.argv[2])
    if len(sys.argv)>3:
        seed = int(sys.argv[3])
    else:
        seed = 0

    np.random.seed(seed)  # fix the seed
    torch.manual_seed(seed)  # fix the seed
    torch.cuda.manual_seed(seed)

    idim = 30  # lift projection dim
    noise_dim = 7  # noisify raw DCA
    split_rate = 0.8
    snr_vals = [10.]  # signal-to-noise ratios
    num_samples = 10000  # samples to collect from the lorenz system

    print("Generating ground truth dynamics ...")
    X_dynamics = gen_lorenz_data(num_samples)  # 10000 * 3
    noisy_model = DNN(X_dynamics.shape[1], idim)  # DNN lift projection: 3 -> 30 for d-DCA

    dca_recons = []
    ddca_recons = []
    r2_vals = np.zeros((len(snr_vals), 2))  # obtain R2 scores for DCA and dDCA
    for snr_idx, snr in enumerate(snr_vals):
        print("Generating noisy data with snr=%.2f ..." % snr)
        X_clean, X_noisy = gen_nonlinear_noisy_lorenz(idim, T, snr, X_dynamics=X_dynamics, noisy_model=noisy_model, seed=seed)
        X_noisy = X_noisy - X_noisy.mean(axis=0)

        X_clean_train, X_clean_val = split(X_clean, split_rate)
        X_noisy_train, X_noisy_val = split(X_noisy, split_rate)
        X_dyn_train, X_dyn_val = split(X_dynamics, split_rate)
        writer = SummaryWriter('runs/ddca_T=%d_reg=%.2f' % (T, ortho_lambda))

        # deep DCA
        use_gpu=True
        if use_gpu:
            device=torch.device("cuda:0")
        print("Training d-DCA")
        ddca_model = DynamicalComponentsAnalysis(idim, fdim=3, T=T, encoder_type="gru", block_toeplitz=False, ortho_lambda=ortho_lambda,
                   dropout=0.5, init="random_ortho")
        # Weiran: chunk long sequences to shorter ones.
        X_train_seqs, L_train = chunk_long_seq(X_noisy_train, 30, 500)
        X_valid_seqs, L_valid = chunk_long_seq(X_noisy_val, 30, 500)
        fit_ddca(ddca_model, X_train_seqs, L_train, X_valid_seqs, L_valid, writer,
                          use_gpu, batch_size=20, max_epochs=50)
        pdb.set_trace()
        X_ddca = ddca_model.encode(torch.from_numpy(X_noisy_val[:500,:]).to(device))
        # print("X_ddca:", X_ddca.shape)
        # ddca_model = opt.model
        # X_ddca = ddca_model(torch.Tensor(X_noisy_val)).detach().cpu().numpy() # reconstruct 3-d signals: X_ddca
        # X_ddca = smoothen(X_ddca)

        # Linear DCA
        print("Training DCA")
        opt = DCA(T=T, d=3, use_scipy=False, block_toeplitz=False, ortho_lambda=10., init="random_ortho",
                  max_epochs=2000)
        opt.fit(X_noisy_train, X_noisy_val, X_dyn_val, writer)
        V_dca = opt.coef_  # transformation matrix
        X_dca = np.dot(X_noisy_val, V_dca)  # recontructed 3-d signals: X_dca
        # X_dca = smoothen(X_dca)

        # match DCA with ground-truth
        print("Matching DCA")
        X_dca_recon = match(X_dca, X_dyn_val, 15000)
        # match d-DCA with ground-truth
        print("Matching d-DCA")
        X_ddca_recon = match(X_ddca.detach().cpu(), X_dyn_val, 15000)

        # R2 of dca
        r2_dca = 1 - np.sum((X_dca_recon - X_dyn_val) ** 2) / np.sum((X_dyn_val - np.mean(X_dyn_val, axis=0)) ** 2)
        # R2 of ddca
        r2_ddca = 1 - np.sum((X_ddca_recon - X_dyn_val) ** 2) / np.sum((X_dyn_val - np.mean(X_dyn_val, axis=0)) ** 2)
        # store R2's
        r2_vals[snr_idx] = [r2_dca, r2_ddca]
        # store reconstructed signals    
        dca_recons.append(X_dca_recon)
        ddca_recons.append(X_ddca_recon)

    plot_figs(dca_recons, ddca_recons, X_dyn_val, X_clean_val, X_noisy_val, r2_vals, snr_vals, "DCA", "d-DCA",
              "figs/fig1_T=%d_reg=%.2f.pdf".format(T, ortho_lambda))
