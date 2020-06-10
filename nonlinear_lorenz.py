import sys
sys.path.append(".")
sys.path.append("..")

import scipy, h5py
import numpy as np
import matplotlib.pyplot as plt

from nl_dca import DynamicalComponentsAnalysis as dDCA, style
from nl_dca.cov_util import calc_cross_cov_mats_from_data
from nl_dca.synth_data import embedded_lorenz_cross_cov_mats, nl_embedded_lorenz_cross_cov_mats, gen_lorenz_data, random_basis, median_subspace
from nl_dca.plotting import lorenz_fig_axes, plot_3d, plot_lorenz_3d, plot_traces, plot_dca_demo, plot_r2, plot_cov
from models import DNN, Match_DNN

from dca import DynamicalComponentsAnalysis as DCA
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn
import torch.nn.functional as F

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

#Set parameters
T = int(sys.argv[1]) # time window
N = 30 # lift projection dim
noise_dim = 7 # noisify raw DCA
#snr_vals = np.logspace(-2, 2, 20)
snr_vals = [10.] # signal-to-noise ratios

RESULTS_FILENAME = "results/lorenz_results_T-{}.hdf5".format(T) # store the results s.t. one can customize the plotting later

def plot_3d_sig(X_dca, plt_idx=300, fig_name="X_dca.png"): # plot 3-d signals in one figure
    idxs = np.arange(plt_idx)
    plt.plot(idxs, X_dca[:plt_idx, 0], color='b')
    plt.plot(idxs, X_dca[:plt_idx, 1], color='r')
    plt.plot(idxs, X_dca[:plt_idx, 2], color='g')
    plt.savefig("figs/{}".format(fig_name))
    plt.clf()

seed=19940423
np.random.seed(seed) # fix the seed
torch.manual_seed(seed) # fix the seed

#Save params
with h5py.File(RESULTS_FILENAME, "w") as f:
    f.attrs["T"] = T
    f.attrs["N"] = N
    f.attrs["noise_dim"] = noise_dim
    f.attrs["snr_vals"] = snr_vals

    #Generate Lorenz dynamics
    num_samples = 10000 # samples to collect from the lorenz system
    X_dynamics = gen_lorenz_data(num_samples) # 10000 *  3
    dynamics_var = np.max(scipy.linalg.eigvalsh(np.cov(X_dynamics.T))) # largest eigenvalue of raw 3-d signals

    #Save dynamics
    f.create_dataset("X_dynamics", data=X_dynamics)
    f.attrs["dynamics_var"] = dynamics_var

    V_dynamics = random_basis(N, 3, np.random) # orthognal lift projection: 3 -> 30 for DCA. V_dynamics contains 3 orthognal 30-d vectors
    
    noisy_model = DNN(X_dynamics.shape[1], N) # DNN lift projection: 3 -> 30 for d-DCA
    X = noisy_model(torch.Tensor(X_dynamics)).detach().numpy() # lift raw signals
    X_var = np.max(scipy.linalg.eigvalsh(np.cov(X.T))) # compute max eigenvalue after lifting
    X *= np.sqrt(dynamics_var/X_var) # match the largest eigenvalue

    #Generate a subspace with median principal angles w.r.t. dynamics subspace. In practice, it is almost equivalent to using V_noise = random_basis(N, N, np.random)
    V_noise = median_subspace(N, noise_dim, num_samples=5000, V_0=V_dynamics, rng=np.random)
    #... and extend V_noise to a basis for R^N
    V_noise_comp = scipy.linalg.orth(np.eye(N) - np.dot(V_noise, V_noise.T))
    V_noise = np.concatenate((V_noise, V_noise_comp), axis=1)

    #Save embeded dynamics and embedding matrices
    f.create_dataset("X", data=X) 
    f.attrs["V_dynamics"] = V_dynamics
    f.attrs["V_noise"] = V_noise

    #To-save: noisy data, reconstructed PCA, reconstructed DCA
    X_noisy_dset = f.create_dataset("X_noisy", (len(snr_vals), num_samples, N))
    X_ddca_trans_dset = f.create_dataset("X_ddca_trans", (len(snr_vals), int(num_samples*0.2), 3))
    X_dca_trans_dset = f.create_dataset("X_dca_trans", (len(snr_vals), int(num_samples*0.2), 3))

    #Loop over SNR vals
    for snr_idx in range(len(snr_vals)):
        snr = snr_vals[snr_idx]
        print("snr =", snr)
       
        _, X_noisy = nl_embedded_lorenz_cross_cov_mats(N, T, snr, noise_dim, return_samples=True, V_dynamics=V_dynamics, V_noise=V_noise, X_dynamics=X_dynamics, model=noisy_model) # non-linearly noisify the raw 3-d signals. output: 30-d noisy signals
        #_, X_noisy = embedded_lorenz_cross_cov_mats(N, T, snr, noise_dim, return_samples=True, V_dynamics=V_dynamics, V_noise=V_noise, X_dynamics=X_dynamics) # linearly noisify the raw 3-d signals. output: 30-d noisy signals

        X_noisy = X_noisy - X_noisy.mean(axis=0) # subtract the mean to ensure the 0 mean
        
        X_noisy_dset[snr_idx] = X_noisy

        # split train and val set
        X_noisy_train = X_noisy[:int(num_samples*0.8), :]
        X_noisy_val = X_noisy[int(num_samples*0.8):, :]
        X_dyn_train = X_dynamics[:int(num_samples*0.8), :]
        X_dyn_val = X_dynamics[int(num_samples*0.8):, :]

        writer = SummaryWriter('runs/ddca_T-{}'.format(T))

        # Run linear DCA
        opt = DCA(T=T, d=3, use_scipy=False, block_toeplitz=False, ortho_lambda=10., init="random_ortho")
        opt.fit(X_noisy_train, X_noisy_val, X_dyn_val, writer)
        V_dca = opt.coef_ # transformation matrix
        X_dca = np.dot(X_noisy_val, V_dca) # recontructed 3-d signals: X_dca

        # Run deep DCA
        opt = dDCA(T=T, d=3, use_scipy=False, block_toeplitz=False, ortho_lambda=0., smooth_lambda=0., init="random_ortho")
        opt.fit(torch.Tensor(X_noisy_train), torch.Tensor(X_noisy_val), torch.Tensor(X_dyn_val), writer)
        dca_model = opt.model
        #dca_model.fc1.weight.data = torch.Tensor(scipy.linalg.orth(dca_model.fc1.weight.data.t().detach().numpy())).t()
        #dca_model.fc2.weight.data = torch.Tensor(scipy.linalg.orth(dca_model.fc2.weight.data.t().detach().numpy())).t()
        X_ddca = dca_model(torch.Tensor(X_noisy_val)).detach().cpu().numpy() # reconstruct 3-d signals: X_ddca

        plot_3d_sig(X_dca, plt_idx=600, fig_name="X_dca_T-{}.png".format(T))
        plot_3d_sig(X_ddca, plt_idx=600, fig_name="X_ddca_T-{}.png".format(T))
        
        #Linearly trasnform projected data to be close to original Lorenz attractor
        '''beta_pca = np.linalg.lstsq(X_pca, X_dynamics, rcond=None)[0]
        beta_dca = np.linalg.lstsq(X_dca, X_dynamics, rcond=None)[0]
        X_pca_trans = np.dot(X_pca, beta_pca)
        X_dca_trans = np.dot(X_dca, beta_dca)'''
        print("match DCA")
        X_dca_trans = match(X_dca, X_dyn_val, 15000)
        print("match dDCA")
        X_ddca_trans = match(X_ddca, X_dyn_val, 15000)

        # plot reconstructed Lorenz attractor and the ground-truth
        plot_3d_sig(X_dca_trans, plt_idx=600, fig_name="X_dca_trans_T-{}.png".format(T))
        plot_3d_sig(X_ddca_trans, plt_idx=600, fig_name="X_ddca_trans_T-{}.png".format(T))
        plot_3d_sig(X_dyn_val, plt_idx=600, fig_name="X_true.png")

        #Save transformed projections
        X_dca_trans_dset[snr_idx] = X_dca_trans
        X_ddca_trans_dset[snr_idx] = X_ddca_trans

with h5py.File(RESULTS_FILENAME, "r") as f:
    snr_vals = f.attrs["snr_vals"][:]
    X = f["X"][:]
    X_noisy_dset = f["X_noisy"][:]
    X_ddca_trans_dset = f["X_ddca_trans"][:]
    X_dca_trans_dset = f["X_dca_trans"][:]
    X_dynamics = f["X_dynamics"][:]

    r2_vals = np.zeros((len(snr_vals), 2)) # obtain R2 scores for DCA and dDCA
    for snr_idx in range(len(snr_vals)):
        X_ddca_trans = X_ddca_trans_dset[snr_idx]
        X_dca_trans = X_dca_trans_dset[snr_idx]
        r2_ddca = 1 - np.sum((X_ddca_trans - X_dyn_val)**2)/np.sum((X_dyn_val - np.mean(X_dyn_val, axis=0))**2)
        r2_dca = 1 - np.sum((X_dca_trans - X_dyn_val)**2)/np.sum((X_dyn_val - np.mean(X_dyn_val, axis=0))**2)
        r2_vals[snr_idx] = [r2_ddca, r2_dca]

#Create axes
axes, txt_cords = lorenz_fig_axes(fig_width=5.5,
                                  wpad_edge=0.01, wpad_mid=0.05,
                                  left_ax_width=0.125, left_ax_wpad=0.025,
                                  hpad_bottom=0.132, hpad_top=0.06, hpad_mid=0.075)
left_ax_1, left_ax_2, left_ax_3, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes
linewidth_3d = 0.5
linewidth_2d = 0.75
linewidth_r2_plot = 1.0

noise_color = "#1261A0"
sig_color = "#3B9CDD"
past_color = "0.85"
future_color = "0.65"
dca_color = "#CF2F25"
ddca_color = "black"

T_to_show_2d = 150
T_to_show_3d = 600
X_display_idx = 0 #index of noisy X dataset to show (make sure to change if SNR spacing changes)

#ax1: Lorenz 3D Plot
plot_lorenz_3d(ax1, X_dyn_val[:T_to_show_3d], linewidth_3d)

#ax2 and ax3: Plots of noiseless and noisy embeddings
N_to_show = 5 #number of channels to plot (also plot last one)
plot_traces(ax2, X[:T_to_show_2d], N_to_show, linewidth_2d)
#divide by a factor to make it look better
plot_traces(ax3, X_noisy_dset[X_display_idx, :T_to_show_2d]/1.8, N_to_show, linewidth_2d)

#ax4 and ax5: Plots of projections (dDCA and random)
#get a random projection of X_noisy and transorm for Lorenz comparison
to_proj = X_noisy_dset[X_display_idx]
np.random.seed(34)
X_random = np.dot(to_proj, scipy.stats.ortho_group.rvs(to_proj.shape[1])[:, :3])
beta_random = np.linalg.lstsq(X_random, X_dynamics, rcond=None)[0]
X_random_trans = np.dot(X_random, beta_random)
plot_dca_demo(ax4, ax5, X_random_trans[:T_to_show_2d], X_ddca_trans_dset[X_display_idx, :T_to_show_2d],
              past_color=past_color, future_color=future_color, linewidth=linewidth_2d)

#Plot Lorenz panels (Qualitative results of 3d projection)
dca_axes = [ax6, ax8, ax10]
ddca_axes = [ax7, ax9, ax11]
#plt_snr_vals = [0.1, 1.0, 10.0]
#plt_snr_strs = ["$10^{-1}$", "$10^{0}$", "$10^{1}$"]
plt_snr_vals = [10.0]
plt_snr_strs = ["$10^{1}$"]
plt_idx = [np.argmin((snr_vals-snr)**2) for snr in plt_snr_vals]
for i in range(len(plt_snr_vals)):
    plot_3d(X_dca_trans_dset[plt_idx[i], :T_to_show_3d], ax=dca_axes[i], color=dca_color, linewidth=linewidth_3d)
    plot_3d(X_ddca_trans_dset[plt_idx[i], :T_to_show_3d], ax=ddca_axes[i], color=ddca_color, linewidth=linewidth_3d)
    dca_axes[i].set_title("SNR = " + plt_snr_strs[i], pad=5, fontsize=style.axis_label_fontsize)
for ax in dca_axes + ddca_axes:
    ax.set_axis_off()
    ax.dist = 7.5
plt.gcf().text(txt_cords[0][0], txt_cords[0][1], "DCA", va="center", ha="center", fontsize=style.axis_label_fontsize, color=dca_color)
plt.gcf().text(txt_cords[1][0], txt_cords[1][1], "d-DCA", va="center", ha="center", fontsize=style.axis_label_fontsize, color=ddca_color)

#Finally, the R2 vs SNR plot
plot_r2(ax12, snr_vals, plt_snr_vals, r2_vals, dca_color, ddca_color)

#Left cov plots
left_ax_1.set_zorder(1000)
plot_cov(left_ax_1, sig_var=1, noise_var=5, noise_sig_labels=True, noise_color=noise_color, sig_color=sig_color, pca_color=ddca_color, dca_color=dca_color)
plot_cov(left_ax_2, sig_var=5, noise_var=5, noise_sig_labels=False, noise_color=noise_color, sig_color=sig_color, pca_color=ddca_color, dca_color=dca_color)
plot_cov(left_ax_3, sig_var=5, noise_var=1, noise_sig_labels=False, noise_color=noise_color, sig_color=sig_color, pca_color=ddca_color, dca_color=dca_color)

plt.savefig("figs/fig1_T-{}.pdf".format(T))
