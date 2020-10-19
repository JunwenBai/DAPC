# Copyright 2020 Salesforce Research (Junwen Bai, Weiran Wang)
# Licensed under the Apache License, Version 2.0 (the "License")

import sys
sys.path.append(".")
sys.path.append("..")

import h5py, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dapc import analysis, data_util
from dapc.dapc import DAPC

import torch
import seaborn as sns
import argparse
from distutils.util import strtobool

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj", default="det", type=str,
                        choices=["det", "cpc", "vae"],
                        help="objective function for representation learning, det (deterministic), cpc, or vae")
    parser.add_argument("--dataset", default="HC", type=str, help="dataset")
    parser.add_argument("--fdim", default=3, help="Dimensionality of features", type=int)
    parser.add_argument("--T", default=4, help="Time steps for estimating PI", type=int)
    parser.add_argument("--ortho_lambda", default=0.0, help="Regularization parameter for orthogonality", type=float)
    parser.add_argument("--recon_lambda", default=0.0, help="Regularization parameter for reconstruction", type=float)
    parser.add_argument("--rate_lambda", default=0.0, help="Regularization parameter for latent space matching", type=float)
    parser.add_argument("--lr", default=1e-3, help="Learning rate", type=float)
    parser.add_argument("--dropout", default=0.0, help="Dropout probability of networks.", type=float)
    parser.add_argument("--batchsize", default=20, help="Number of sequences in each minibatch", type=int)
    parser.add_argument("--encoder_type", default="lin", type=str, choices=["lin", "transformer", "dnn", "gru", "lstm", "bgru", "blstm"])
    parser.add_argument("--base_encoder_type", default="lin", type=str, choices=["lin", "dnn", "gru", "lstm", "bgru", "blstm"])
    parser.add_argument("--epochs", default=10, help="Number of training epochs", type=int)
    parser.add_argument('--masked_recon', default=False, type=strtobool, help='Whether to use masked reconstruction loss')
    parser.add_argument("--gpuid", default="0", help="ID of gpu device to be used", type=str)
    parser.add_argument("--seed", default=0, help="Random seed", type=int)
    return parser

def main(args):
    parser = get_parser()
    parser = DAPC.add_arguments(parser)
    args = parser.parse_args(args)

    ### data
    M1 = data_util.load_sabes_data('data/neural/indy_20160627_01.mat')
    HC = data_util.load_kording_paper_data('data/neural/example_data_hc.pickle')

    #### params
    T_pi_vals = np.arange(1, 11)
    dims = np.array([15])

    offsets = np.array([0, 5, 10, 15])

    win = 3 # decoding window size
    n_cv = 5 # num of cross validation
    n_init = 5
    ####

    if args.dataset == "HC":
        print("HC:", HC['neural'].shape, HC['loc'].shape) # inputs and targets
        if not os.path.exists("results"):
            os.mkdir("results")
        res_name = "results/HC_{}_recon{}_ortho{}_{}.npy".format(args.obj, args.recon_lambda, args.ortho_lambda, args.epochs)

        HC_results = analysis.run_analysis(HC['neural'], HC['loc'], T_pi_vals, dim_vals=dims, offset_vals=offsets, res_name=res_name,
                                       num_cv_folds=n_cv, decoding_window=win, args=args, n_init=n_init, verbose=True, index=0)
        np.save(res_name, HC_results)
    elif args.dataset == "M1":
        print("M1:", M1['M1'].shape, M1['cursor'].shape)
        if not os.path.exists("results"):
            os.mkdir("results")
        res_name = "results/M1_{}_recon{}_ortho{}_{}.npy".format(args.obj, args.recon_lambda, args.ortho_lambda, args.epochs)

        M1_results = analysis.run_analysis(M1['M1'], M1['cursor'], T_pi_vals, dim_vals=dims, offset_vals=offsets, res_name=res_name,
                                       num_cv_folds=n_cv, decoding_window=win, args=args, n_init=n_init, verbose=True, index=0)
        np.save(res_name, M1_results)


if __name__ == "__main__":
    main(sys.argv[1:])

