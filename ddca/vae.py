import torch
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from .math import log_density_gaussian, log_importance_weight_matrix, matrix_log_density_gaussian
import pdb


def vdca_loss_junwen(latent_dist, latent_sample, latent_mask, T, cov, post_L, alpha=0., beta=0., gamma=1., zeta=1.):
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
    # Weiran: below is Junwen's implementation.
    # block_kl_loss = torch.sum((block_log_qz - block_log_pz) * mask) / torch.sum(mask)

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


"""
Weiran: Older stuff not in use.
def btcvae_loss(latent_dist, latent_sample, is_mss=True, alpha=1., beta=6.):
    batch_size, seq_len, latent_dim = latent_sample.shape

    #print("latent_dist:", latent_dist[0].shape, latent_dist[1].shape)
    #print("latent_sample:", latent_sample.shape)
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, is_mss=is_mss)

    # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    mi_loss = (log_q_zCx - log_qz).mean()
    # TC[z] = KL[q(z)||\prod_i z_i]
    tc_loss = (log_qz - log_prod_qzi).mean()
    # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
    dw_kl_loss = (log_prod_qzi - log_pz).mean()
    # anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal) if is_train else 1)
    #print("mi_loss:", mi_loss.item())
    #print("tc_loss:", tc_loss.item())
    #print("dw_kl_loss:", dw_kl_loss.item())

    # total loss
    loss = alpha * mi_loss + beta * tc_loss
    return loss


def vdca_rate_loss(latent_dist, latent_sample, hmask, T, cov):
    latent_mu, latent_logvar = latent_dist # 20*500*3
    batch_size, seq_len, d = latent_sample.shape
    # hmask is of shape (batch_size, seq_len)

    # calculate log q(z|x)
    latent_sample = latent_sample.view(batch_size, seq_len*d).unfold(1, 2 * T * d, d).contiguous()  # [20, 493, 24]
    latent_mu = latent_mu.view(batch_size, seq_len*d).unfold(1, 2 * T * d, d).contiguous()
    latent_logvar = latent_logvar.view(batch_size, seq_len*d).unfold(1, 2 * T * d, d).contiguous()
    hmask_unfold = hmask.float().unfold(1, 2 * T, 1).contiguous().mean(-1)
    #log_q_zCx = log_density_gaussian(latent_sample.view(-1, d), latent_mu.view(-1, d), latent_logvar.view(-1, d))
    log_q_zCx = log_density_gaussian(latent_sample, latent_mu, latent_logvar).sum(dim=2) # 20*493

    # calculate log p(z)
    mvn = MVN(torch.zeros(2 * T * d, device=cov.device), covariance_matrix=cov)
    log_pz = mvn.log_prob(latent_sample) # 20*493

    # Here I am using a naive implementation of KL.
    # rates = torch.exp(log_q_zCx) * (log_q_zCx - log_pz)
    rates = log_q_zCx - log_pz
    return torch.sum(rates * hmask_unfold) / torch.sum(hmask_unfold)
"""