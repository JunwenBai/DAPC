# Copyright 2020 Salesforce Research (Junwen Bai, Weiran Wang)
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.decomposition import PCA
from scipy.stats import special_ortho_group as sog

from .cov_utils import calc_cov_from_data
from .data_util import CrossValidate, form_lag_matrix
from .dapc import DAPC, fit_dapc
from .data_process import match, split, chunk_long_seq, smoothen
from .utils import linear_decode_r2

import torch
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_analysis(X, Y, T_pi_vals, dim_vals, offset_vals, res_name, num_cv_folds, decoding_window, args,
                 n_init=1, verbose=False, index=1):

    # X: 1363 * 30
    # Y: 1363 * 30
    use_gpu = True
    device = torch.device("cuda:{}".format(args.gpuid))

    results_size = (num_cv_folds, len(dim_vals), len(offset_vals), len(T_pi_vals) + 2)
    results = np.zeros(results_size) # 5*4*4*12
    min_std = 1e-6
    good_cols = (X.std(axis=0) > min_std)
    X = X[:, good_cols]
    # loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds, stack=False)
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        if fold_idx: break
        if verbose:
            print("fold", fold_idx + 1, "of", num_cv_folds)

        # mean-center X and Y
        X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        X_train_ctd = [Xi - X_mean for Xi in X_train]
        X_test_ctd = X_test - X_mean
        Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        Y_test_ctd = Y_test - Y_mean

        # chunk
        chunk_size = 500
        X_train_seqs, L_train = chunk_long_seq(X_train_ctd[0], 30, chunk_size)
        X_test_seqs, L_test = [X_test_ctd], [X_test_ctd.shape[0]]

        # compute cross-cov mats for DCA
        T_max = 2 * np.max(T_pi_vals)

        # loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            if verbose:
                print("dim", dim_idx + 1, "of", len(dim_vals))

            # loop over T_pi vals
            for T_pi_idx in range(len(T_pi_vals)):

                T_pi = T_pi_vals[T_pi_idx]

                idim = X_test_ctd.shape[-1]
                fdim = dim
                T = T_pi
                params = 'obj={}_encoder={}_fdim={}_context={}_T={}_lr={}_bs={}_dropout={}_rate-lambda={}_ortho-lambda={}_recon-lambda={}_seed={}'.format(
                    args.obj, args.encoder_type, args.fdim, args.input_context, args.T, args.lr, args.batchsize, args.dropout, args.rate_lambda, args.ortho_lambda, args.recon_lambda, args.seed)

                dapc_model = DAPC(args.obj, idim, fdim, T, encoder_type=args.encoder_type,
                                                 ortho_lambda=args.ortho_lambda, recon_lambda=args.recon_lambda,
                                                 dropout=args.dropout, masked_recon=args.masked_recon,
                                                 args=args, device=device)

                dapc_model = fit_dapc(dapc_model, X_train_seqs, L_train, X_test_seqs, L_test, None, args.lr, use_gpu,
                             batch_size=args.batchsize, max_epochs=args.epochs, device=device, snapshot=params + ".cpt", use_writer=False, pred_data=[X_train_ctd[0], Y_train_ctd[0], X_test_ctd, Y_test_ctd, decoding_window, offset_vals[-1], args])

                # compute DCA R2 over offsets
                X_train_dapc = dapc_model.encode(torch.from_numpy(X_train_ctd[0]).to(device, dtype=dapc_model.dtype)).cpu().numpy()
                X_test_dapc = dapc_model.encode(torch.from_numpy(X_test_ctd).to(device, dtype=dapc_model.dtype)).cpu().numpy()
                for offset_idx in range(len(offset_vals)):
                    offset = offset_vals[offset_idx]
                    r2_dapc = linear_decode_r2(X_train_dapc, Y_train_ctd, X_test_dapc, Y_test_ctd,
                                              decoding_window=decoding_window, offset=offset)
                    results[fold_idx, dim_idx, offset_idx, T_pi_idx] = r2_dapc
                    np.save(res_name, results)

    return results

