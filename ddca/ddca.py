import numpy as np
import math
import logging

import torch
import torch.nn.functional as F
from .solver import LIN, DNN, RNN, TRANSFORMER, ortho_reg_fn, ortho_reg_Y
from .cov_utils import calc_cov_from_data, calc_pi_from_cov, matrix_toeplitzify
from .utils import make_non_pad_mask, pad_list, _context_concat, gen_batch_indices
from .spec_augment import spectral_masking
from .vae import vdca_loss_junwen
from distutils.util import strtobool
import pdb
from .data_process import match

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
        group.add_argument('--vae_ndata', default=1, type=int,
                           help='number of samples drawn from posterior')
        group.add_argument('--vae_alpha', default=0., type=float,
                           help='alpha')
        group.add_argument('--vae_beta', default=1., type=float,
                           help='beta')
        group.add_argument('--vae_gamma', default=0., type=float,
                           help='gamma')
        group.add_argument('--vae_zeta', default=0., type=float,
                           help='zeta')
        parser.add_argument('--use_prior_pi', default=False, type=strtobool,
                            help='Whether to compute pi on the prior')
        parser.add_argument('--use_dim_pi', default=False, type=strtobool,
                            help='Whether to use dim-wise pi')
        parser.add_argument('--vae_pseudo_utts', default=1, type=int,
                            help='Number of pseudo utterance')
        parser.add_argument('--vae_pseudo_maxlen', default=500, type=int,
                            help='Number of frames for pseudo inputs')

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
        group.add_argument('--block_toeplitz', default=True, type=strtobool,
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

    def __init__(self, obj, idim, fdim, T, encoder_type, ortho_lambda, recon_lambda, dropout,
            masked_recon, args, dtype=torch.float32, device="cuda:0"):
        super(DynamicalComponentsAnalysis, self).__init__()

        # Objective for representation learning.
        self.obj = obj

        # Splicing options.
        self.input_context = args.input_context
        self.idim = idim * (1+2*self.input_context)
        self.fdim = fdim

        # Model params.
        self.device = device
        self.T = T
        self.recon_lambda = recon_lambda
        self.ortho_lambda = ortho_lambda
        self.rate_lambda = args.rate_lambda
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.dtype = dtype
        self.block_toeplitz = args.block_toeplitz
        self.cov_diag_reg = args.cov_diag_reg

        # Masked reconstruction params.
        self.masked_recon = masked_recon
        self.spec_mask_F = args.spec_mask_F
        self.spec_mask_T = args.spec_mask_T
        self.num_freq_masks = args.num_freq_masks
        self.num_time_masks = args.num_time_masks

        # VAE params
        self.vae_alpha = args.vae_alpha
        self.vae_beta = args.vae_beta
        self.vae_gamma = args.vae_gamma
        self.vae_zeta = args.vae_zeta
        self.use_prior_pi = args.use_prior_pi
        self.use_dim_pi = args.use_dim_pi
        if self.obj == "vae":
            print("VAE coeffs: alpha=%f, beta=%f, gamma=%f, zeta=%f" % (self.vae_alpha, self.vae_beta, self.vae_gamma, self.vae_zeta))

        # CPC params.
        self.cpc_num_pos = args.cpc_num_pos
        self.cpc_num_neg = args.cpc_num_neg

        if self.obj == "vae":
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

        if self.obj == "cpc":
            self.cpc_future_enc = LIN(self.idim, self.fdim, dropout=self.dropout)
        else:
            self.cpc_future_enc = None

        # Weiran: the prior has zero mean and covariance L*L' (to ensure PSD).
        # self.vae_prior_L = torch.nn.Parameter(torch.eye(2 * self.fdim * self.T, dtype=self.dtype).to(device), requires_grad=True)

        # The pseudo input approach.
        self.vae_pseudo_utts = args.vae_pseudo_utts
        self.vae_pseudo_maxlen = args.vae_pseudo_maxlen
        self.pseudo_lens = None
        self.pseudo_inputs = torch.nn.Parameter(torch.zeros([args.vae_pseudo_utts, args.vae_pseudo_maxlen, self.idim], dtype=self.dtype).uniform_(0, 1).to(device), requires_grad=True)
        self.vae_posterior_L = torch.nn.Linear(2 * self.fdim * self.T, 2 * self.fdim * self.T)

        # Weiran: based on my experience, reconstruction network would better be a DNN than RNNs.
        if self.recon_lambda > 0:
            self.decoder = DNN(self.fdim, self.idim, h_sizes=[args.encoder_dnn_hidden_size] * args.encoder_dnn_num_layers, dropout=self.dropout)
        else:
            self.decoder = None

    def set_pseudo_inputs(self, pseudo_inputs, pseudo_lens):
        """Initialize parameters."""
        with torch.no_grad():
            assert pseudo_inputs.shape[0] == self.pseudo_inputs.shape[0], "pseudo inputs number mismatch"
            assert pseudo_inputs.shape[1] == self.pseudo_inputs.shape[1], "pseudo inputs length mismatch"
            assert pseudo_inputs.shape[2] == self.pseudo_inputs.shape[2], "pseudo inputs dimension mismatch"
            print("Initializing pseudo inputs!")
            self.pseudo_inputs.copy_(pseudo_inputs)
            self.pseudo_lens = pseudo_lens

    def vae_split(self, hs_pad, split_size):

        logvar = hs_pad[:, :, split_size:].contiguous()
        mu = hs_pad[:, :, :split_size].contiguous()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        samples = mu + eps * std
        
        return mu, logvar, samples


    def cpc_latent(self, xs_pad, hs_pad, ilens):

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
        hs_pad, olens = self.encoder(xs_pad, ilens)
        hmask = make_non_pad_mask(olens.tolist()).to(xs_pad.device)

        # Let us change the ortho_loss into latent loss, as they are loss in the feature space.
        if self.obj == "vae":
            # Samples are then used for reconstruction.
            mu, logvar, hs_pad = self.vae_split(hs_pad, self.fdim)

        if self.decoder:
            if not self.masked_recon:
                # Normal reconstruction.
                assert (masks_in is None) and (masks_out is None), "regular reconstruction requires no in/out masks"
                recon = self.decoder(hs_pad)
                loss = torch.sum((recon.view([-1, self.idim]) - xs_pad.view([-1, self.idim])) ** 2, 1)
                mask_float = hmask.float().view([-1])
                recon_loss = torch.sum(torch.mul(loss, mask_float)) / torch.sum(mask_float)
            else:
                # Masked reconstruction.
                assert (masks_in is not None) and (masks_out is not None), "masked reconstruction requires nonzero in/out masks"
                hs_pad1, _ = self.encoder(xs_pad * masks_in, ilens)
                if self.obj == "vae":
                    # Samples are now used for reconstruction.
                    hs_mu1, _, hs_pad1 = self.vae_split(hs_pad1, self.fdim)
                    #hs_pad1 = hs_mu1
                recon_loss = torch.sum(((self.decoder(hs_pad1) - xs_pad) ** 2) * masks_out) / torch.sum(masks_out)
        else:
            recon_loss = 0.0
        
        # Weiran: always compute per-frame cov matrix and monitor the ortho_loss.
        ortho_loss, self.cov_frame = ortho_reg_Y(hs_pad, hmask)
        if self.obj == "vae":
            _, self.mu_cov_frame = ortho_reg_Y(mu, hmask)
        else:
            self.mu_cov_frame = self.cov_frame
        if self.encoder_type == "lin" and not self.obj == "vae":
            # This is a special setting that mimics the original DCA.
            ortho_loss = ortho_reg_fn(self.encoder.fc1.weight.t())

        # Weiran: always compute the 2T-time cov matrix and monitor the pi.
        if self.obj == "vae":
            # self.cov = torch.mm(self.vae_prior_L, self.vae_prior_L.t()) + self.cov_diag_reg * torch.eye(2 * self.T * self.fdim).to(self.device)
            # self.cov = matrix_toeplitzify(self.cov, 2*self.T, self.fdim) + self.cov_diag_reg * torch.eye(2 * self.T * self.fdim).to(self.device)
            pseudo_feats, pseudo_olens = self.encoder(self.pseudo_inputs, self.pseudo_lens)
            pseudo_hmask = make_non_pad_mask(pseudo_olens.tolist()).to(xs_pad.device)
            pseudo_samples, _, _ = self.vae_split(pseudo_feats, self.fdim)
            self.cov = calc_cov_from_data(pseudo_samples, pseudo_hmask, 2 * self.T, toeplitzify=False, reg=self.cov_diag_reg)

            self.post_cov = calc_cov_from_data(hs_pad, hmask, 2 * self.T, toeplitzify=self.block_toeplitz, reg=self.cov_diag_reg)
        else:
            self.cov = calc_cov_from_data(hs_pad, hmask, 2 * self.T, toeplitzify=self.block_toeplitz, reg=self.cov_diag_reg)

        if not self.obj == "vae":
            # For deterministic methods, not using dim pi.
            pi = calc_pi_from_cov(self.cov)
        else:
            if self.use_dim_pi:
                pi = 0.
                weights = torch.ones(self.fdim)
                for i in range(self.fdim):
                    if not self.use_prior_pi:
                        sub_pi = calc_pi_from_cov(self.post_cov[i::self.fdim, i::self.fdim]) * weights[i]
                    else:
                        sub_pi = calc_pi_from_cov(self.cov[i::self.fdim, i::self.fdim]) * weights[i]
                    pi += sub_pi
            else:
                if self.use_prior_pi:
                    pi = calc_pi_from_cov(self.cov)
                else:
                    pi = calc_pi_from_cov(self.post_cov)

        # Compile the total loss.
        rate_loss = 0.
        if self.obj == "cpc":
            # Since pi and ortho_loss do not make sense for CPC.
            key_loss = self.cpc_latent(xs_pad, hs_pad, ilens)
        elif self.obj == "vae":
            rate_loss = vdca_loss_junwen((mu, logvar), hs_pad, hmask, self.T, self.cov, self.vae_posterior_L,
                                    alpha=self.vae_alpha, beta=self.vae_beta, gamma=self.vae_gamma, zeta=self.vae_zeta)
            key_loss = -pi + self.rate_lambda * rate_loss  #+ 0.1 * self.cov.norm(2)
        else:
            # For deterministic method, we use pi and ortho_loss.
            key_loss = -pi + self.ortho_lambda * ortho_loss

        self.loss = key_loss + self.recon_lambda * recon_loss
        return self.loss, float(pi), float(ortho_loss), float(recon_loss), float(rate_loss), (self.cov_frame).detach().cpu().numpy(), (self.mu_cov_frame).detach().cpu().numpy()


    def encode(self, x):
        # Weiran: encode only one utterance.
        # x is a 2D tensor of shape time x idim.
        self.eval()
        ilens = torch.tensor([x.size(0)], device=x.device).long()
        hs_pad, _ = self.encoder(x.unsqueeze(0), ilens)
        if self.obj == "vae":
            hs_pad, _, _ = self.vae_split(hs_pad, self.fdim)
        return hs_pad.squeeze(0).detach()


# Move training code out of model definition.
def fit_ddca(model, X_train, L_train, X_valid, L_valid, writer, lr=1e-3, use_gpu=False, batch_size=50, max_epochs=500,
            device="cuda:0", snapshot='ddca_snapshot', X_match=None, Y_match=None, use_writer=True):
    if use_gpu:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    input_context = model.input_context

    # X_train is a sequence of input sequences, whose lengths are saved in L_train.
    # The proper way is to write a data loader to get next batch.
    n_train = len(X_train)
    n_batch_train = int(math.ceil(n_train / batch_size))
    n_valid = len(X_valid)
    n_batch_valid = int(math.ceil(n_valid / batch_size))

    best_mse = np.inf

    for epoch in range(max_epochs):
        model.train()
        order = np.random.permutation(n_train)
        total_loss = 0.0
        total_pi = 0.0
        total_ortho_loss = 0.0
        total_loss_recon = 0.0
        total_loss_rate = 0.0
        
        for i in range(n_batch_train):
            idx_batch = list(order[i * batch_size: min((i + 1) * batch_size, n_train)])

            x_batch_list = [torch.from_numpy(_context_concat(X_train[_], input_context)).float() for _ in idx_batch]
            l_batch_list = [L_train[_] for _ in idx_batch]
            total_len_batch = sum(l_batch_list)
            x_batch, l_batch = pad_list(x_batch_list, 0.0).to(device), torch.Tensor(l_batch_list).long().to(device)

            if epoch==0 and i==0:
                model.set_pseudo_inputs(x_batch[:model.vae_pseudo_utts, :model.vae_pseudo_maxlen, :], l_batch[:model.vae_pseudo_utts])

            optimizer.zero_grad()
            if model.recon_lambda > 0 and model.masked_recon:
                # Weiran: move these steps into a separate function.
                masks_in = [spectral_masking(torch.ones_like(x), F=model.spec_mask_F, T=model.spec_mask_T,
                    num_freq_masks=model.num_freq_masks, num_time_masks=model.num_time_masks).numpy() for x in x_batch_list]
                masks_out = [1.0 -m for m in masks_in]
                masks_in = [_context_concat(m, input_context) for m in masks_in]
                masks_out = [_context_concat(m, input_context) for m in masks_out]
                masks_in = pad_list([torch.from_numpy(x).float() for x in masks_in], 0).to(device)
                masks_out = pad_list([torch.from_numpy(x).float() for x in masks_out], 0).to(device)
                loss, pi, loss_orth, loss_recon, loss_rate, cov_frame, cov = model(x_batch, l_batch, masks_in, masks_out)
            else:
                loss, pi, loss_orth, loss_recon, loss_rate, cov_frame, cov = model(x_batch, l_batch)

            # loss, pi, loss_orth, loss_recon, _ = model(x_batch, l_batch)
            #print("minibatch %03d/%03d: pi=%f" % (i, n_batch_train, pi))
            loss.backward()
            """
            total_norm = 0.
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1/2.)
            print("total_norm:", total_norm)
            """
            loss.detach()
            optimizer.step()

            total_loss += loss
            total_pi += pi
            total_ortho_loss += loss_orth
            total_loss_recon += loss_recon
            total_loss_rate += loss_rate

        avg_loss_train = total_loss / n_batch_train
        avg_pi_train = total_pi / n_batch_train
        avg_ortho_loss_train = total_ortho_loss / n_batch_train
        avg_recon_loss_train = total_loss_recon / n_batch_train
        avg_rate_loss_train = total_loss_rate / n_batch_train
        print("epoch %d, train avg loss=%f, pi=%f, ortho_loss=%f, recon_loss=%f, rate_loss=%f" %
              (epoch, avg_loss_train, avg_pi_train, avg_ortho_loss_train, avg_recon_loss_train, avg_rate_loss_train))

        model.eval()
        total_loss = 0.0
        total_pi = 0.0
        total_ortho_loss = 0.0
        total_loss_recon = 0.0
        total_loss_rate = 0.0
        total_cov_frame = np.zeros([model.fdim, model.fdim])
        #total_cov = np.zeros([2*model.fdim*model.T, 2*model.fdim*model.T])
        total_cov = np.zeros([model.fdim, model.fdim])

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
                loss, pi, loss_orth, loss_recon, loss_rate, cov_frame, cov = model(x_batch, l_batch, masks_in, masks_out)
            else:
                loss, pi, loss_orth, loss_recon, loss_rate, cov_frame, cov = model(x_batch, l_batch)
            #print("minibatch %03d/%03d: pi=%f" % (i, n_batch_valid, pi))

            total_loss += loss.detach()
            total_pi += pi
            total_ortho_loss += loss_orth
            total_loss_recon += loss_recon
            total_loss_rate += loss_rate
            total_cov_frame += cov_frame
            total_cov += cov

        avg_loss_valid = total_loss / n_batch_valid
        avg_pi_valid = total_pi / n_batch_valid
        avg_ortho_loss_valid = total_ortho_loss / n_batch_valid
        avg_recon_loss_valid = total_loss_recon / n_batch_valid
        avg_rate_loss_valid = total_loss_rate / n_batch_valid
        avg_cov_frame = total_cov_frame / n_batch_valid
        avg_cov = total_cov / n_batch_valid
        print("epoch %d, valid avg loss=%f, pi=%f, ortho_loss=%f, recon_loss=%f, rate_loss=%f" %
              (epoch, avg_loss_valid, avg_pi_valid, avg_ortho_loss_valid, avg_recon_loss_valid, avg_rate_loss_valid))
        print(avg_cov_frame)
        print(avg_cov)

        mse = evaluate_match(model, X_match, Y_match, verbose=0)
        if mse < best_mse:
            best_mse = mse
            print("*** Updating best model! ***")
            torch.save(model.state_dict(), "best_" + snapshot)

        # Write stats.
        if use_writer:
            writer.add_scalar('train/pi', avg_pi_train, epoch)
            writer.add_scalar('train/orth', avg_ortho_loss_train, epoch)
            writer.add_scalar('train/recon', avg_recon_loss_train, epoch)
            writer.add_scalar('train/rate', avg_rate_loss_train, epoch)
            writer.add_scalar('valid/pi', avg_pi_valid, epoch)
            writer.add_scalar('valid/orth', avg_ortho_loss_valid, epoch)
            writer.add_scalar('valid/recon', avg_recon_loss_valid, epoch)
            writer.add_scalar('valid/rate', avg_rate_loss_valid, epoch)
            writer.add_scalar('valid/match_mse', mse, epoch)

    print("Resuming from best snapshot ...")
    model.load_state_dict(torch.load("best_" + snapshot))
    return model

def evaluate_match(model, X_match, Y_match, verbose=1):
    Y_pred = model.encode(X_match)
    recon, mse = match(Y_pred, Y_match, 15000, model.device, verbose=verbose)
    return mse

