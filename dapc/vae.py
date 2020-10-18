# Copyright 2020 Salesforce Research (Junwen Bai, Weiran Wang)
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from .math import log_density_gaussian, log_importance_weight_matrix, matrix_log_density_gaussian
import pdb


def vdapc_loss(latent_dist, latent_sample, latent_mask, T, cov, post_L, alpha=0., beta=0., gamma=1., zeta=1.):
    batch_size, seq_len, d = latent_sample.shape

    ### Junwen: compute the log prob terms for each 3*3 block
    # Weiran: sample across different utts.
    latent_mu = latent_dist[0].reshape(-1, d)
    latent_logvar = latent_dist[1].reshape(-1, d)
    mask = latent_mask.reshape(-1).float()
    # This gives indices of valid samples.
    idx = mask.nonzero()[:, 0]
    if idx.shape[0] > 2000:
        step = idx.shape[0] // 2000
        latent_mu_sub = latent_mu[idx[::step], :]
        latent_logvar_sub = latent_logvar[idx[::step], :]
    else:
        latent_mu_sub = latent_mu[idx, :]
        latent_logvar_sub = latent_mu[idx, :]

    block_log_pz, block_log_qz, block_log_prod_qzi, block_log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
        latent_sample.reshape(-1, d), (latent_mu, latent_logvar), (latent_mu_sub, latent_logvar_sub))
    block_mi_loss = torch.sum((block_log_q_zCx - block_log_qz) * mask) / torch.sum(mask)
    block_tc_loss = torch.sum((block_log_qz - block_log_prod_qzi) * mask) / torch.sum(mask)
    block_kl_loss = torch.sum((block_log_prod_qzi - block_log_pz) * mask) / torch.sum(mask)

    ### Junwen: compute the log prob terms for each 24*24 block
    latent_sample_2T = latent_sample.reshape(batch_size, seq_len*d).unfold(1, 2*T*d, d).reshape(-1, 2*T*d)
    latent_mu = latent_mu.reshape(batch_size, seq_len*d).unfold(1, 2*T*d, d).reshape(-1, 2*T*d)
    latent_logvar = latent_logvar.reshape(batch_size, seq_len*d).unfold(1, 2*T*d, d).reshape(-1, 2*T*d)
    mask = latent_mask.reshape(batch_size, seq_len).unfold(1, 2*T, 1).reshape(-1, 2*T).all(1).float()
    log_q_zCx = log_density_gaussian(latent_sample_2T, latent_mu, latent_logvar).sum(1)

    mvn = MVN(torch.zeros(2 * T * d, device=cov.device), covariance_matrix=cov)
    latent_sample_2T = post_L(latent_sample_2T)
    log_pz = mvn.log_prob(latent_sample_2T)
    kl_loss = torch.sum((log_q_zCx - log_pz) * mask) / torch.sum(mask)

    # the choice of the losses could be arbitrary combination of diff terms for diff-size blocks
    loss = alpha * block_mi_loss + beta * block_tc_loss + gamma * block_kl_loss + zeta * kl_loss

    print("vae losses: block_mi_loss=%f, block_tc_loss=%f, block_kl_loss=%f, kl_loss=%f" %
          (block_mi_loss, block_tc_loss, block_kl_loss, kl_loss))
    return loss


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, latent_dist_sub, is_mss=False, n_data=None):

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist_sub)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

