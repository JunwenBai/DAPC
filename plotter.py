import torch
import torch.nn as nn
import numpy as np
import scipy
from ddca.plotting import lorenz_fig_axes, plot_3d, plot_lorenz_3d, plot_traces, plot_dca_demo, plot_r2, plot_cov
import matplotlib.pyplot as plt

def plot_figs(dca_recons, ddca_recons, X_dyn_val, X_clean_val, X_noisy_val, r2_vals, snr_vals, label1="DCA", label2="d-DCA", fig_name="figs/fig1.pdf"):
	#Create axes
	dca_recons, ddca_recons, snr_vals = np.array(dca_recons), np.array(ddca_recons), np.array(snr_vals)
	axes, txt_cords = lorenz_fig_axes(fig_width=5.5,
									  wpad_edge=0.01, wpad_mid=0.05,
									  left_ax_width=0.125, left_ax_wpad=0.025,
									  hpad_bottom=0.132, hpad_top=0.06, hpad_mid=0.075)
	left_ax_1, left_ax_2, left_ax_3, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes
	fontsize=8
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
	T_to_show_3d = 500
	X_display_idx = 0 #index of noisy X dataset to show (make sure to change if SNR spacing changes)

	#ax1: Lorenz 3D Plot
	plot_lorenz_3d(ax1, X_dyn_val[:T_to_show_3d], linewidth_3d)

	#ax2 and ax3: Plots of noiseless and noisy embeddings
	N_to_show = 5 #number of channels to plot (also plot last one)
	plot_traces(ax2, X_clean_val[:T_to_show_2d], N_to_show, linewidth_2d)
	#divide by a factor to make it look better
	plot_traces(ax3, X_noisy_val[:T_to_show_2d], N_to_show, linewidth_2d)

	#ax4 and ax5: Plots of projections (dDCA and random)
	#get a random projection of X_noisy and transorm for Lorenz comparison
	to_proj = X_noisy_val
	X_random = np.dot(to_proj, scipy.stats.ortho_group.rvs(to_proj.shape[1])[:, :3])
	beta_random = np.linalg.lstsq(X_random, X_dyn_val, rcond=None)[0]
	X_random_trans = np.dot(X_random, beta_random)
	plot_dca_demo(ax4, ax5, X_random_trans[:T_to_show_2d], ddca_recons[X_display_idx, :T_to_show_2d],
				  past_color=past_color, future_color=future_color, linewidth=linewidth_2d, label=label2)
	
	#Plot Lorenz panels (Qualitative results of 3d projection)
	dca_axes = [ax6, ax8, ax10]
	ddca_axes = [ax7, ax9, ax11]
	#plt_snr_vals = [0.1, 1.0, 10.0]
	#plt_snr_strs = ["$10^{-1}$", "$10^{0}$", "$10^{1}$"]
	plt_snr_vals = [10.0]
	plt_snr_strs = ["$10^{1}$"]
	plt_idx = [np.argmin((snr_vals-snr)**2) for snr in plt_snr_vals]
	for i in range(len(plt_snr_vals)):
		plot_3d(dca_recons[plt_idx[i], :T_to_show_3d], ax=dca_axes[i], color=dca_color, linewidth=linewidth_3d)
		plot_3d(ddca_recons[plt_idx[i], :T_to_show_3d], ax=ddca_axes[i], color=ddca_color, linewidth=linewidth_3d)
		dca_axes[i].set_title("SNR = " + plt_snr_strs[i], pad=5, fontsize=fontsize)
	for ax in dca_axes + ddca_axes:
		ax.set_axis_off()
		ax.dist = 7.5
	plt.gcf().text(txt_cords[0][0], txt_cords[0][1], label1, va="center", ha="center", fontsize=fontsize, color=dca_color)
	plt.gcf().text(txt_cords[1][0], txt_cords[1][1], label2, va="center", ha="center", fontsize=fontsize, color=ddca_color)

	#Finally, the R2 vs SNR plot
	plot_r2(ax12, snr_vals, plt_snr_vals, r2_vals, dca_color, ddca_color, label1, label2)

	#Left cov plots
	left_ax_1.set_zorder(1000)
	plot_cov(left_ax_1, sig_var=1, noise_var=5, noise_sig_labels=True, noise_color=noise_color, sig_color=sig_color, pca_color=ddca_color, dca_color=dca_color)
	plot_cov(left_ax_2, sig_var=5, noise_var=5, noise_sig_labels=False, noise_color=noise_color, sig_color=sig_color, pca_color=ddca_color, dca_color=dca_color)
	plot_cov(left_ax_3, sig_var=5, noise_var=1, noise_sig_labels=False, noise_color=noise_color, sig_color=sig_color, pca_color=ddca_color, dca_color=dca_color)

	plt.savefig(fig_name)
