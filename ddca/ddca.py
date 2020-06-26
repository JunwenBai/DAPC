import numpy as np
import math
import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .solver import LIN, DNN, RNN, TRANSFORMER, ortho_reg_fn, ortho_reg_Y
from .cov_utils import calc_cov_from_data, calc_pi_from_cov
from .utils import make_non_pad_mask, pad_list, _context_concat, gen_batch_indices
from .spec_augment import spectral_masking
from .vae import btcvae_loss, vdca_loss, vdca_rate_loss
from distutils.util import strtobool
import pdb


class DynamicalComponentsAnalysis(torch.nn.Module):
    """Dynamical Components Analysis.

    Runs DCA on multidimensional time series data X to discover a projection
    onto a fdim-dimensional subspace which maximizes the complexity, as defined by the Gaussian
    Predictive Information (PI) of the fdim-dimensional dynamics over windows of length T.

    Parameters
    ----------
    idim : int
        Sequence's input dimensionality.
    fdim : int
        Number of basis vectors onto which the data X are projected.
    T : int
        Size of time windows across which to compute mutual information. Total window length will be
        `2 * T`. When fitting a model, the length of the shortest time series must be greater than
        `2 * T` and for good performance should be much greater than `2 * T`.
    ortho_lambda : float
        Coefficient on term that keeps projected covariance close to identity.
    recon_lambda : float
        Coefficient on reconstruction loss term.
    dtype : pytorch.dtype
        What dtype to use for computation.
    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("DynamicalComponentsAnalysis model setting")

        # Weiran: transformer-related.
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="linear",
                           choices=["linear", "embed"],
                           help='transformer input layer type')
        # Weiran: death rates for stochastic layers.
        group.add_argument('--edeath-rate', default=0.0, type=float,
                           help='death rate for encoder')
        # Encoder
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')

        # VAE-related
        group.add_argument('--vae_alpha', default=0., type=float,
                           help='alpha')
        group.add_argument('--vae_beta', default=0., type=float,
                           help='beta')
        group.add_argument('--vae_gamma', default=1., type=float,
                           help='gamma')
        group.add_argument('--vae_zeta', default=1., type=float,
                           help='zeta')

        # CPC-related.
        group.add_argument('--cpc_num_pos', default=4, type=int,
                           help='Number of positive samples for CPC')
        group.add_argument('--cpc_num_neg', default=16, type=int,
                           help='Number of negative samples for CPC')

        # Specaugment-related.
        group.add_argument('--spec_mask_F', default=5, type=int,
                           help='Maximum width of frequency masks')
        group.add_argument('--spec_mask_T', default=40, type=int,
                           help='Maximum width of time masks')
        group.add_argument('--num_freq_masks', default=2, type=int,
                           help='Number of frequency masks')
        group.add_argument('--num_time_masks', default=2, type=int,
                           help='Number of time masks')

        # Covariance regularization.
        group.add_argument('--block_toeplitz', default=False, type=strtobool,
                           help='Whether to Toeplitzify the covariance matrix')
        group.add_argument('--cov_diag_reg', default=1e-6, type=float,
                           help='Constants added to the diagonal of covariance matrix')

        # Input.
        group.add_argument('--input_context', default=0, type=int,
                           help='Number of left and right frames used for splicing')

        # Encoder architecture.
        group.add_argument('--encoder_rnn_num_layers', default=3, type=int,
                           help='Number hidden layers for encoder RNN')
        group.add_argument('--encoder_rnn_hidden_size', default=256, type=int,
                           help='Number of hidden units for encoder RNN')

        group.add_argument('--encoder_dnn_num_layers', default=3, type=int,
                           help='Number hidden layers for encoder DNN')
        group.add_argument('--encoder_dnn_hidden_size', default=512, type=int,
                           help='Number of hidden units for encoder DNN')

        return parser

    def __init__(self, idim, fdim, T, encoder_type, ortho_lambda, recon_lambda, dropout,
            use_cpc, masked_recon, args, use_vae=False, n_data=None, dtype=torch.float32, device="cuda:0"):
        super(DynamicalComponentsAnalysis, self).__init__()

        # Splicing options.
        self.input_context = args.input_context
        self.idim = idim * (1+2*self.input_context)
        self.fdim = fdim
        self.n_data = n_data

        # Model params.
        self.device = device
        self.T = T
        self.recon_lambda = recon_lambda
        self.ortho_lambda = ortho_lambda
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.dtype = dtype
        self.block_toeplitz = args.block_toeplitz
        self.cov_diag_reg = args.cov_diag_reg

        '''
        cholesky = torch.tril((torch.rand(2*T*fdim, 2*T*fdim)*2-1.)*math.sqrt(0.1/(2.*T*fdim)))+torch.eye(2*T*fdim)
        #cholesky = torch.eye(2*T*fdim)
        #cholesky = torch.tril(torch.ones(2*T*fdim, 2*T*fdim))
        #hs_cov = torch.load("pts/cov.pt")
        self.chol = cholesky.to(device).clone().detach().requires_grad_(True)
        #self.chol = torch.matmul(cholesky, cholesky.t()).to(device).clone().detach().requires_grad_(True)
        #self.mu = torch.load("pts/hs_mean.pt").to(device).clone().detach().requires_grad_(True)
        self.mu = torch.zeros(2*T*fdim).to(device).clone().detach().requires_grad_(True)
        '''

        '''self.chol = torch.nn.Parameter(cholesky)
        self.chol.requires_grad = True
        self.mu = torch.nn.Parameter(torch.zeros(2*T*fdim))
        self.mu.requires_grad = True'''
        '''self.cov_mask = torch.zeros(2*T*fdim, 2*T*fdim).to(device)
        for i in range(2*T*fdim):
            for j in range(2*T*fdim):
                if abs(i-j) % fdim == 0:
                    self.cov_mask[i, j] = 1.
        self.cov_mask.requires_grad_(False)
        self.chol = (torch.matmul(cholesky, cholesky.t()).to(device)+self.cov_mask).clone().detach().requires_grad_(True)'''

        # Masked reconstruction params.
        self.masked_recon = masked_recon
        self.spec_mask_F = args.spec_mask_F
        self.spec_mask_T = args.spec_mask_T
        self.num_freq_masks = args.num_freq_masks
        self.num_time_masks = args.num_time_masks

        # VAE params
        self.use_vae = use_vae
        self.vae_alpha = args.vae_alpha
        self.vae_beta = args.vae_beta
        self.vae_gamma = args.vae_gamma
        self.vae_zeta = args.vae_zeta
        if self.use_vae:
            print("VAE coeffs:", self.vae_alpha, self.vae_beta, self.vae_gamma, self.vae_zeta)

        # CPC params.
        self.use_cpc = use_cpc
        self.cpc_num_pos = args.cpc_num_pos
        self.cpc_num_neg = args.cpc_num_neg

        if use_vae:
            # In case of VAE, we need both mean and variance.
            self.encoder_odim = self.fdim * 2
        else:
            self.encoder_odim = self.fdim

        if self.encoder_type == "lin":
            self.encoder = LIN(self.idim, self.encoder_odim, dropout=self.dropout)
        elif self.encoder_type == "dnn":
            self.encoder = DNN(self.idim, self.encoder_odim,
                    h_sizes=[args.encoder_dnn_hidden_size] * args.encoder_dnn_num_layers, dropout=self.dropout)
        elif self.encoder_type == "transformer":
            self.encoder = TRANSFORMER(
                idim=idim * (1+2*self.input_context),
                odim=self.encoder_odim,
                adim=args.adim,
                aheads=args.aheads,
                eunits=args.eunits,
                elayers=args.elayers,
                input_layer=args.transformer_input_layer,
                dropout_rate=self.dropout,
                death_rate=args.edeath_rate)
        else:
            # Choices are ['lstm', 'gru', 'blstm', 'bgru'].
            self.encoder = RNN(idim=self.idim, elayers=args.encoder_rnn_num_layers, cdim=args.encoder_rnn_hidden_size,
                    hdim=self.encoder_odim, dropout=self.dropout, typ=self.encoder_type)

        if self.use_cpc:
            self.cpc_future_enc = LIN(self.idim, self.fdim, dropout=self.dropout)
        else:
            self.cpc_future_enc = None

        if self.use_vae:
            # Weiran: the prior has zero mean and covariance L*L' (to ensure PSD).
            self.vae_prior_L = Variable(torch.eye(2 * self.fdim * self.T).type(self.dtype), requires_grad=True)
        else:
            self.vae_prior_L = None

        # Weiran: based on my experience, reconstruction network would better be a DNN than RNNs.
        if self.recon_lambda > 0:
            self.decoder = DNN(self.fdim, self.idim, h_sizes=[args.encoder_dnn_hidden_size] * args.encoder_dnn_num_layers, dropout=self.dropout)
        else:
            self.decoder = None


    def vae_split(self, hs_pad, split_size):

        logvar = hs_pad[:, :, split_size:].contiguous()
        mu = hs_pad[:, :, :split_size].contiguous()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        samples = mu + eps * std

        return mu, logvar, samples


    def cpc_latent(self, xs_pad, hs_pad, hmask, ilens):

        slfidx, posidx, negidx = gen_batch_indices(ilens, max(ilens), range(self.cpc_num_pos), self.cpc_num_neg, portion=1.0)
        fx = hs_pad.view(-1, hs_pad.shape[-1])
        gy = self.cpc_future_enc(xs_pad.view(-1, xs_pad.shape[-1]))
        fx_slf = fx[slfidx]
        gy_pos = gy[posidx]
        pos_score = torch.sum(fx_slf * gy_pos, 1, keepdim=True)
        fx_slf = fx_slf.repeat(1, self.cpc_num_neg).view(-1, self.fdim)
        gy_neg = gy[negidx]
        neg_score = torch.sum(fx_slf * gy_neg, 1).view(-1, self.cpc_num_neg)
        scores = torch.cat([pos_score, neg_score], 1)
        log_preds = F.log_softmax(scores)
        cpc_loss = - torch.mean(log_preds[:, 0])
        return cpc_loss


    def forward(self, xs_pad, ilens, masks_in=None, masks_out=None):
        """ forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, maxlen, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :return: total loss value
        :rtype: torch.Tensor
        :return: PI loss value
        :rtype: float
        :return: regularization value
        :rtype: float
        """

        # Weiran: for now, both RNN and TRANSFORMER do not reduce the input lengths.
        hs_pad, olens, _ = self.encoder(xs_pad, ilens)
        hmask = make_non_pad_mask(olens.tolist()).to(xs_pad.device)

        # Let us change the ortho_loss into latent loss, as they are loss in the feature space.
        if self.use_vae:
            # Samples are then used for reconstruction.
            mu, logvar, hs_pad = self.vae_split(hs_pad, self.fdim)

        if self.decoder:
            if not self.masked_recon:
                # Normal reconstruction.
                assert (masks_in is None) and (masks_out is None), "regular reconstruction requires no in/out masks"
                recon = self.decoder(hs_pad)
                # recon_loss = compute_recon_mse(recon, xs_pad, hmask)
                loss = torch.sum((recon.view([-1, self.idim]) - xs_pad.view([-1, self.idim])) ** 2, 1)
                mask_float = hmask.float().view([-1])
                recon_loss = torch.sum(torch.mul(loss, mask_float)) / torch.sum(mask_float)
            else:
                # Masked reconstruction.
                assert (masks_in is not None) and (masks_out is not None), "masked reconstruction requires nonzero in/out masks"
                hs_pad1, _ = self.encoder(xs_pad * masks_in, ilens)
                if self.use_vae:
                    # Samples are now used for reconstruction.
                    _, _, hs_pad1 = self.vae_split(hs_pad1, self.fdim)
                recon_loss = torch.sum(((self.decoder(hs_pad1) - xs_pad) ** 2) * masks_out) / torch.sum(masks_out)
        else:
            recon_loss = 0.0
        
        #if self.use_vae:
        #    hs_pad = hs_pad_mu

        # Weiran: always compute per-frame cov matrix and monitor the ortho_loss.
        ortho_loss, self.cov_frame = ortho_reg_Y(hs_pad, hmask)
        if self.encoder_type == "lin" and not self.use_vae:
            # This is a special setting that mimics the original DCA.
            ortho_loss = ortho_reg_fn(self.encoder.fc1.weight.t())

        # Weiran: always compute the 2T-time cov matrix and monitor the pi.
        if self.use_vae:
            self.cov = torch.mm(self.vae_prior_L, self.vae_prior_L.t()) + self.cov_diag_reg * torch.eye(2 * self.T * self.fdim)
        else:
            self.cov = calc_cov_from_data(hs_pad, hmask, 2 * self.T, toeplitzify=self.block_toeplitz, reg=self.cov_diag_reg)
        pi = calc_pi_from_cov(self.cov)

        # Compile the total loss.
        if self.use_cpc:
            # Since pi and ortho_loss do not make sense for CPC.
            key_loss = self.cpc_latent(xs_pad, hs_pad, hmask, ilens)
        elif self.use_vae:
            '''
            # kld = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # latent_loss = btcvae_loss((mu, logvar), samples, n_data=self.n_data, is_mss=False, alpha=1., beta=20.)
            latent_loss = vdca_loss((mu, logvar), hs_pad, hmask, self.mu, self.vae_prior_L, self.T, n_data=self.n_data,
                                    alpha=self.vae_alpha, beta=self.vae_beta, gamma=self.vae_gamma, zeta=self.vae_zeta)
            '''
            rate_loss = vdca_rate_loss((mu, logvar), hs_pad, hmask, self.T, self.cov)
            # Weiran: according "Fixing a broken ELBO" and beta-VAE, the beta is ratio between rate and log-likelihood.
            key_loss = -pi + self.vae_beta * self.recon_lambda * rate_loss
        else:
            # For deterministic method, we use pi and ortho_loss.
            key_loss = -pi + self.ortho_lambda * ortho_loss

        self.loss = key_loss + self.recon_lambda * recon_loss
        return self.loss, float(pi), float(ortho_loss), float(recon_loss), (self.cov_frame).detach().cpu().numpy(), (self.cov).detach().cpu().numpy()


    def encode(self, x):
        # Weiran: encode only one utterance.
        # x is a 2D tensor of shape time x idim.
        self.eval()
        ilens = torch.tensor([x.size(0)], device=x.device).long()
        hs_pad, _ = self.encoder(x.unsqueeze(0), ilens)
        if self.use_vae:
            hs_pad, _, _ = self.vae_split(hs_pad, self.fdim)
        return hs_pad.squeeze(0).detach()


# Move training code out of model definition.
def fit_ddca(model, X_train, L_train, X_valid, L_valid, writer, use_gpu=False, batch_size=50, max_epochs=500, device="cuda:0"):
    if use_gpu:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim_prior = torch.optim.Adam([model.chol, model.mu], lr=1e-3)
    input_context = model.input_context

    # X_train is a sequence of input sequences, whose lengths are saved in L_train.
    # The proper way is to write a data loader to get next batch.
    n_train = len(X_train)
    n_batch_train = int(math.ceil(n_train / batch_size))
    n_valid = len(X_valid)
    n_batch_valid = int(math.ceil(n_valid / batch_size))

    for epoch in range(max_epochs):
        model.train()
        order = np.random.permutation(n_train)
        total_loss = 0.0
        total_pi = 0.0
        total_ortho_loss = 0.0
        total_loss_recon = 0.0
        
        for i in range(n_batch_train):
            idx_batch = list(order[i * batch_size: min((i + 1) * batch_size, n_train)])

            x_batch_list = [torch.from_numpy(_context_concat(X_train[_], input_context)).float() for _ in idx_batch]
            l_batch_list = [L_train[_] for _ in idx_batch]
            total_len_batch = sum(l_batch_list)
            x_batch, l_batch = pad_list(x_batch_list, 0.0).to(device), torch.Tensor(l_batch_list).long().to(device)

            optimizer.zero_grad()
            optim_prior.zero_grad()
            if model.recon_lambda > 0 and model.masked_recon:
                # Weiran: move these steps into a separate function.
                masks_in = [spectral_masking(torch.ones_like(x), F=model.spec_mask_F, T=model.spec_mask_T,
                    num_freq_masks=model.num_freq_masks, num_time_masks=model.num_time_masks).numpy() for x in x_batch_list]
                masks_out = [1.0 -m for m in masks_in]
                masks_in = [_context_concat(m, input_context) for m in masks_in]
                masks_out = [_context_concat(m, input_context) for m in masks_out]
                masks_in = pad_list([torch.from_numpy(x).float() for x in masks_in], 0).to(device)
                masks_out = pad_list([torch.from_numpy(x).float() for x in masks_out], 0).to(device)
                loss, pi, loss_orth, loss_recon, cov_frame = model(x_batch, l_batch, masks_in, masks_out)
            else:
                loss, pi, loss_orth, loss_recon, cov_frame = model(x_batch, l_batch)

            # loss, pi, loss_orth, loss_recon, _ = model(x_batch, l_batch)
            #print("minibatch %03d/%03d: pi=%f" % (i, n_batch_train, pi))
            loss.backward()
            loss.detach()
            optimizer.step()
            optim_prior.step()
            #model.chol = model.chol * cov_mask
            #model.chol = (model.chol+model.chol.t()) / 2.

            total_loss += loss
            total_pi += pi
            total_ortho_loss += loss_orth
            total_loss_recon += loss_recon * total_len_batch

        print(torch.matmul(model.chol, model.chol.t())[:6, :6])
        #print(model.chol[:6, :6])

        avg_loss_train = total_loss / n_batch_train
        avg_pi_train = total_pi / n_batch_train
        avg_ortho_loss_train = total_ortho_loss / n_batch_train
        avg_recon_loss_train = total_loss_recon / sum(L_train)
        print("epoch %d, train avg loss=%f, pi=%f, ortho_loss=%f, recon_loss=%f" %
              (epoch, avg_loss_train, avg_pi_train, avg_ortho_loss_train, avg_recon_loss_train))

        model.eval()
        total_loss = 0.0
        total_pi = 0.0
        total_ortho_loss = 0.0
        total_loss_recon = 0.0
        total_cov_frame = np.zeros([model.fdim, model.fdim])
        for i in range(int(math.ceil(n_valid / batch_size))):
            x_batch_list = [torch.from_numpy(_context_concat(X_valid[_],input_context)).float() for _ in range(i*batch_size, min((i+1)*batch_size, n_valid))]
            l_batch_list = [L_valid[_] for _ in range(i*batch_size, min((i+1)*batch_size, n_valid))]
            total_len_batch = sum(l_batch_list)
            x_batch, l_batch = pad_list(x_batch_list, 0.0).to(device), torch.Tensor(l_batch_list).long().to(device)

            if model.recon_lambda > 0 and model.masked_recon:
                # Weiran: move these steps into a separate function.
                masks_in = [spectral_masking(torch.ones_like(x), F=model.spec_mask_F, T=model.spec_mask_T,
                    num_freq_masks=model.num_freq_masks, num_time_masks=model.num_time_masks).numpy() for x in x_batch_list]
                masks_out = [1.0 -m for m in masks_in]
                masks_in = [_context_concat(m, input_context) for m in masks_in]
                masks_out = [_context_concat(m, input_context) for m in masks_out]
                masks_in = pad_list([torch.from_numpy(x).float() for x in masks_in], 0).to(device)
                masks_out = pad_list([torch.from_numpy(x).float() for x in masks_out], 0).to(device)
                loss, pi, loss_orth, loss_recon, cov_frame = model(x_batch, l_batch, masks_in, masks_out)
            else:
                loss, pi, loss_orth, loss_recon, cov_frame = model(x_batch, l_batch)
            #print("minibatch %03d/%03d: pi=%f" % (i, n_batch_valid, pi))

            total_loss += loss.detach()
            total_pi += pi
            total_ortho_loss += loss_orth
            total_loss_recon += loss_recon * total_len_batch
            total_cov_frame += cov_frame * total_len_batch

        avg_loss_valid = total_loss / n_batch_valid
        avg_pi_valid = total_pi / n_batch_valid
        avg_ortho_loss_valid = total_ortho_loss / n_batch_valid
        avg_recon_loss_valid = total_loss_recon / sum(L_valid)
        avg_cov_frame = total_cov_frame / sum(L_valid)
        print("epoch %d, valid avg loss=%f, pi=%f, ortho_loss=%f, recon_loss=%f" %
              (epoch, avg_loss_valid, avg_pi_valid, avg_ortho_loss_valid, avg_recon_loss_valid))
        print(avg_cov_frame)

        # Write stats.
        writer.add_scalar('train/pi', avg_pi_train, epoch)
        writer.add_scalar('train/orth', avg_ortho_loss_train, epoch)
        writer.add_scalar('train/recon', avg_recon_loss_train, epoch)
        writer.add_scalar('valid/pi', avg_pi_valid, epoch)
        writer.add_scalar('valid/orth', avg_ortho_loss_valid, epoch)
        writer.add_scalar('valid/recon', avg_recon_loss_valid, epoch)

    return model
