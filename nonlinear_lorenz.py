import sys
sys.path.append(".")
sys.path.append("..")

import scipy, h5py
import numpy as np
import matplotlib.pyplot as plt

from ddca import DynamicalComponentsAnalysis as dDCA
from ddca.data_gen import gen_nonlinear_noisy_lorenz, gen_lorenz_data
from models.DNN import DNN, Match_DNN
from plotter import plot_figs
from dca import DynamicalComponentsAnalysis as DCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def smoothen(raw_xs,window_len=12,window='hamming'):
    xs = raw_xs.T
    ys = []
    for x in xs:
        if window_len<3:
            return x
        s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.'+window+'(window_len)')
        y = np.convolve(w/w.sum(),s,mode='valid')
        y = y[window_len//2:-((window_len-1)//2)]
        ys.append(y)
    ys = np.array(ys).T
    return ys

def match(X, X_true, max_epochs=3000): # use a linear mapping to match the reconstructed lorenz attractor and the ground truth attractor
    if not isinstance(X, torch.Tensor): X = torch.Tensor(X) # torch tensorize
    if not isinstance(X_true, torch.Tensor): X_true = torch.Tensor(X_true) # torch tensorize

    match_model = Match_DNN(X.shape[1], X_true.shape[1]) # a linear model for matching
    match_opt = torch.optim.Adam(match_model.parameters(), lr=1e-3) # Adam for optimizing
    for epoch in range(max_epochs): # one can overfit as much as possible since the optimization is for reconstruction
        X_rec = match_model(X)
        match_opt.zero_grad()
        loss = F.mse_loss(X_rec, X_true) # alternative losses: l1
        loss.backward()
        if epoch % 2000 == 0:
            print(epoch, ":", loss.item())
        match_opt.step()
    
    return match_model(X).detach().cpu().numpy()

def plot_3d_sig(X_dca, plt_idx=300, fig_name="X_dca.png"): # plot 3-d signals in one figure
    idxs = np.arange(plt_idx)
    plt.plot(idxs, X_dca[:plt_idx, 0], color='b')
    plt.plot(idxs, X_dca[:plt_idx, 1], color='r')
    plt.plot(idxs, X_dca[:plt_idx, 2], color='g')
    plt.savefig("figs/{}".format(fig_name))
    plt.clf()

def split(X, split_rate):
    len_X = len(X)
    return X[:int(len_X*split_rate), :], X[int(len_X*split_rate):, :]

if __name__ == "__main__":
    #Set parameters
    T = int(sys.argv[1]) # time window
    seed = int(sys.argv[2])
    np.random.seed(seed) # fix the seed
    torch.manual_seed(seed) # fix the seed
    torch.cuda.manual_seed(seed)

    N = 30 # lift projection dim
    noise_dim = 7 # noisify raw DCA
    split_rate = 0.8
    snr_vals = [10.] # signal-to-noise ratios
    num_samples = 10000 # samples to collect from the lorenz system
    X_dynamics = gen_lorenz_data(num_samples) # 10000 * 3
    noisy_model = DNN(X_dynamics.shape[1], N) # DNN lift projection: 3 -> 30 for d-DCA

    dca_recons = []
    ddca_recons = []
    r2_vals = np.zeros((len(snr_vals), 2)) # obtain R2 scores for DCA and dDCA
    for snr_idx, snr in enumerate(snr_vals):
        X_clean, X_noisy = gen_nonlinear_noisy_lorenz(N, T, snr, X_dynamics=X_dynamics, noisy_model=noisy_model, seed=seed)
        X_noisy = X_noisy - X_noisy.mean(axis=0)

        X_clean_train, X_clean_val = split(X_clean, split_rate)
        X_noisy_train, X_noisy_val = split(X_noisy, split_rate)
        X_dyn_train, X_dyn_val = split(X_dynamics, split_rate)
        writer = SummaryWriter('runs/ddca_T-{}'.format(T))
        
        # deep DCA
        print("Training d-DCA")
        opt = dDCA(T=T, d=3, use_scipy=False, block_toeplitz=False, ortho_lambda=1., smooth_lambda=0., init="random_ortho", max_epochs=201, dropout=0.5, solver="gru", batch_size=1, device="cuda:0")
        X_ddca = opt.fit(torch.Tensor(X_noisy_train), torch.Tensor(X_noisy_val), torch.Tensor(X_dyn_val), writer)
        #print("X_ddca:", X_ddca.shape)
        #ddca_model = opt.model
        #X_ddca = ddca_model(torch.Tensor(X_noisy_val)).detach().cpu().numpy() # reconstruct 3-d signals: X_ddca
        #X_ddca = smoothen(X_ddca)

        # Linear DCA
        print("Training DCA")
        opt = DCA(T=T, d=3, use_scipy=False, block_toeplitz=False, ortho_lambda=10., init="random_ortho", max_epochs=2000)
        opt.fit(X_noisy_train, X_noisy_val, X_dyn_val, writer)
        V_dca = opt.coef_ # transformation matrix
        X_dca = np.dot(X_noisy_val, V_dca) # recontructed 3-d signals: X_dca
        #X_dca = smoothen(X_dca)

        print()
        # match DCA with ground-truth
        print("Matching DCA")
        X_dca_recon = match(X_dca, X_dyn_val, 15000)
        # match d-DCA with ground-truth
        print("Matching d-DCA")
        X_ddca_recon = match(X_ddca.detach().cpu(), X_dyn_val, 15000)

        # R2 of dca
        r2_dca = 1 - np.sum((X_dca_recon - X_dyn_val)**2)/np.sum((X_dyn_val - np.mean(X_dyn_val, axis=0))**2)
        # R2 of ddca
        r2_ddca = 1 - np.sum((X_ddca_recon - X_dyn_val)**2)/np.sum((X_dyn_val - np.mean(X_dyn_val, axis=0))**2)
        # store R2's
        r2_vals[snr_idx] = [r2_dca, r2_ddca]
        # store reconstructed signals    
        dca_recons.append(X_dca_recon)
        ddca_recons.append(X_ddca_recon)

    plot_figs(dca_recons, ddca_recons, X_dyn_val, X_clean_val, X_noisy_val, r2_vals, snr_vals, "DCA", "d-DCA", "figs/fig1_T-{}_seed-{}.pdf".format(T, seed))

