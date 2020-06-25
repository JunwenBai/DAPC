import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from .math import log_density_gaussian, log_importance_weight_matrix, matrix_log_density_gaussian

def vdca_loss(latent_dist, latent_sample, mu, chol, T, n_data, alpha=0., beta=0., gamma=1., zeta=1.):
    batch_size, seq_len, d = latent_sample.shape
    latent_mu, latent_logvar = latent_dist

    ### Junwen: compute the log prob terms for each 3*3 block
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=False)
    block_tc_loss = (log_qz - log_prod_qzi).mean()
    block_mi_loss = (log_q_zCx - log_qz).mean()
    ###

    ### Junwen: compute the log prob terms for each 24*24 block
    log_q_zCx = log_density_gaussian(latent_sample.view(-1, d), latent_mu.view(-1, d), latent_logvar.view(-1, d)).sum(dim=1).view(batch_size, -1).unfold(1, 2*T, 1).sum(2)
    
    latent_sample = latent_sample.view(batch_size, seq_len*d).unfold(1, 2*T*d, d) # [20, 493, 24]
    latent_mu = latent_mu.view(batch_size, seq_len*d).unfold(1, 2*T*d, d)
    latent_logvar = latent_logvar.view(batch_size, seq_len*d).unfold(1, 2*T*d, d)
    mat_log_qz = matrix_log_density_gaussian(latent_sample, latent_mu, latent_logvar) # 20*493*493*24
    log_qz = torch.logsumexp(mat_log_qz.sum(3), dim=2, keepdim=False) # actually log-mean-exp. only diff by log(1/(seq_len))
   
    mvn = MVN(mu, scale_tril=chol)
    log_pz = mvn.log_prob(latent_sample)

    mi_loss = (log_q_zCx - log_qz).mean()
    prior_loss = (log_qz - log_pz).mean()

    # the choice of the losses could be arbitrary combination of diff terms for diff-size blocks
    # if gamma=zeta=0, then vdca_loss is the same as btcvae_loss
    loss = alpha * block_mi_loss + beta * block_tc_loss + gamma * prior_loss + zeta * mi_loss
    return loss

def btcvae_loss(latent_dist, latent_sample, n_data, is_mss=True, alpha=1., beta=6.):
    batch_size, seq_len, latent_dim = latent_sample.shape

    #print("latent_dist:", latent_dist[0].shape, latent_dist[1].shape)
    #print("latent_sample:", latent_sample.shape)
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                         latent_dist,
                                                                         n_data,
                                                                         is_mss=is_mss)
    # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    mi_loss = (log_q_zCx - log_qz).mean()
    # TC[z] = KL[q(z)||\prod_i z_i]
    tc_loss = (log_qz - log_prod_qzi).mean()
    # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
    dw_kl_loss = (log_prod_qzi - log_pz).mean()
    # anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal) if is_train else 1)
    '''print("mi_loss:", mi_loss.item())
    print("tc_loss:", tc_loss.item())
    print("dw_kl_loss:", dw_kl_loss.item())'''

    # total loss
    loss = alpha * mi_loss + beta * tc_loss
    return loss


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, seq_len, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=2)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(2)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    '''if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)'''

    log_qz = torch.logsumexp(mat_log_qz.sum(3), dim=2, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=2, keepdim=False).sum(2)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


