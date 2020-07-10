import sys, os
import pdb

sys.path.append(".")
sys.path.append("..")

import numpy as np
from sklearn.manifold import TSNE
from distutils.util import strtobool

from ddca.ddca import DynamicalComponentsAnalysis
from ddca.ddca import fit_ddca
from ddca.utils import _context_concat, parsegpuid
from ddca.data_gen import gen_nonlinear_noisy_lorenz, gen_lorenz_data
from ddca.data_process import match, split, chunk_long_seq, smoothen
from ddca.solver import LIN, DNN, KERNEL
from ddca.plotting import plot_figs
from dca import DynamicalComponentsAnalysis as Linear_DCA

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj", default="dca", type=str,
                        choices=["dca", "cpc", "vae"],
                        help="objective function for representation learning, dca (deterministic), cpc, or vae")
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

def create_writer_name(writer_path):
    if not os.path.exists(writer_path):
        return writer_path
    cnt = 1
    while os.path.exists(writer_path+"_"+str(cnt)):
        cnt += 1
    return writer_path+"_"+str(cnt)

def main(args):
    parser = get_parser()
    parser = DynamicalComponentsAnalysis.add_arguments(parser)
    args = parser.parse_args(args)

    np.random.seed(args.seed)  # fix the seed
    torch.manual_seed(args.seed)  # fix the seed
    torch.cuda.manual_seed(args.seed)

    # Handle multiple gpu issues.
    gpuid = args.gpuid
    gpulist = parsegpuid(gpuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpulist])
    numGPUs = len(gpulist)
    print("Using %d gpus, CUDA_VISIBLE_DEVICES=%s" % (numGPUs, os.environ["CUDA_VISIBLE_DEVICES"]))

    T = args.T
    fdim = args.fdim
    encoder_name = args.encoder_type
    params = 'obj={}_encoder={}_fdim={}_context={}_T={}_lr={}_bs={}_dropout={}_rate-lambda={}_ortho-lambda={}_recon-lambda={}_seed={}'.format(
			args.obj, encoder_name, args.fdim, args.input_context, args.T, args.lr, args.batchsize, args.dropout, args.rate_lambda, args.ortho_lambda, args.recon_lambda, args.seed)
    if args.obj == "vae":
        params = params + "_priorpi={}_dimpi={}_{}_{}_{}_{}".format(args.use_prior_pi, args.use_dim_pi,  args.vae_alpha, args.vae_beta, args.vae_gamma, args.vae_zeta)
    print(params)

    idim = 30  # lift projection dim
    noise_dim = 7  # noisify raw DCA
    split_rate = 0.82
    snr_vals = [1.]  # signal-to-noise ratios
    num_samples = 10000  # samples to collect from the lorenz system

    print("Generating ground truth dynamics ...")
    X_dynamics = gen_lorenz_data(num_samples)  # 10000 * 3

    noisy_model = DNN(X_dynamics.shape[1], idim, dropout=0.5)  # DNN lift projection: 3 -> 30 for d-DCA
    # noisy_model = KERNEL(X_dynamics[::25], np.linspace(0.3, 1.0, 30))
    # noisy_model = LIN(X_dynamics.shape[1], idim)
    # pdb.set_trace()
    use_gpu = True
    if use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dca_recons = []
    ddca_recons = []
    r2_vals = np.zeros((len(snr_vals), 2))  # obtain R2 scores for DCA and dDCA
    for snr_idx, snr in enumerate(snr_vals):
        print("Generating noisy data with snr=%.2f ..." % snr)
        X_clean, X_noisy = gen_nonlinear_noisy_lorenz(idim, T, snr, X_dynamics=X_dynamics, noisy_model=noisy_model, seed=args.seed)
        X_noisy = X_noisy - X_noisy.mean(axis=0)

        X_clean_train, X_clean_val = split(X_clean, split_rate)
        X_noisy_train, X_noisy_val = split(X_noisy, split_rate)
        X_dyn_train, X_dyn_val = split(X_dynamics, split_rate)
        writer = SummaryWriter(create_writer_name('runs/ddca_{}'.format(params)))
        
        # Weiran: chunk long sequences to shorter ones.
        chunk_size = 500
        X_train_seqs, L_train = chunk_long_seq(X_noisy_train, 30, chunk_size)
        X_valid_seqs, L_valid = chunk_long_seq(X_noisy_val, 30, chunk_size)
        X_clean_seqs, L_clean = chunk_long_seq(X_clean_val, 30, chunk_size)
        X_dyn_seqs, L_dyn = chunk_long_seq(X_dyn_val, 30, chunk_size)
        
        X_match = torch.from_numpy(_context_concat(X_noisy_val[:500], 0)).float().to(device)
        Y_match = X_dyn_val[:500]
        # Linear DCA
        print("Training {}".format(args.base_encoder_type))
        '''
        opt = Linear_DCA(T=T, d=3, use_scipy=False, block_toeplitz=False, ortho_lambda=10., init="random_ortho",
                  max_epochs=1500, device="cpu")
        opt.fit(X_noisy_train, X_noisy_val, X_dyn_val, writer)
        V_dca = opt.coef_  # transformation matrix
        X_dca = np.dot(X_noisy_val, V_dca)  # recontructed 3-d signals: X_dca
        X_dca = X_dca[:500, :]
        X_dca_recon = match(X_dca, X_dyn_val[:500], 15000, device)
        '''
        
        if args.base_encoder_type != "lin":
            dca_model = DynamicalComponentsAnalysis(args.obj, idim, fdim, T, encoder_type=args.base_encoder_type,
                                                    ortho_lambda=args.ortho_lambda, recon_lambda=args.recon_lambda,
                                                    dropout=args.dropout, masked_recon=args.masked_recon,
                                                    args=args, device=device)
        else:
            dca_model = DynamicalComponentsAnalysis("ddca", idim, fdim, T, encoder_type="lin",
                                                    ortho_lambda=10.0, recon_lambda=0.0,
                                                    dropout=0.0, masked_recon=False,
                                                    args=args)
        dca_model = fit_ddca(dca_model, X_train_seqs, L_train, X_valid_seqs[:1], L_valid[:1], writer, args.lr, use_gpu,
                              batch_size=args.batchsize, max_epochs=10, device=device, snapshot="lin_ddca.cpt", X_match=X_match, Y_match=Y_match, use_writer=False)

        X_dca = dca_model.encode(
            torch.from_numpy(_context_concat(X_noisy_val[:500], dca_model.input_context)).float().to(device,
                                                                dtype=dca_model.dtype)).cpu().numpy()
        if X_dca.shape[1] > 3:
            X_dca = TSNE(n_components=3).fit_transform(X_dca)

        # deep DCA
        print("Training {}".format(encoder_name))
        ddca_model = DynamicalComponentsAnalysis(args.obj, idim, fdim, T, encoder_type=args.encoder_type,
                                                 ortho_lambda=args.ortho_lambda, recon_lambda=args.recon_lambda,
                                                 dropout=args.dropout, masked_recon=args.masked_recon,
                                                 args=args, device=device)
     
        ddca_model = fit_ddca(ddca_model, X_train_seqs, L_train, X_valid_seqs, L_valid, writer, args.lr, use_gpu,
                batch_size=args.batchsize, max_epochs=args.epochs, device=device, snapshot=params + ".cpt", X_match=X_match, Y_match=Y_match)

        X_ddca = ddca_model.encode(
            torch.from_numpy(_context_concat(X_noisy_val[:500], ddca_model.input_context)).float().to(device,
                                                            dtype=ddca_model.dtype)).cpu().numpy()
        if X_ddca.shape[1] > 3:
            X_ddca = TSNE(n_components=3).fit_transform(X_ddca)
        #print(X_ddca)
        print(np.matmul((X_ddca - X_ddca.mean(0)).T, (X_ddca - X_ddca.mean(0))) / X_ddca.shape[0])
        
        if ddca_model.obj == "vae":
            ax = sns.heatmap(ddca_model.post_cov.detach().cpu().numpy(), linewidth=0.05)
            plt.savefig("pngs/post_cov_heat_{}.png".format(params))
            plt.clf()
            ax = sns.heatmap(ddca_model.cov.detach().cpu().numpy(), linewidth=0.05)
            plt.savefig("pngs/cov_heat_{}.png".format(params))
        else:
            ax = sns.heatmap(ddca_model.cov.detach().cpu().numpy(), linewidth=0.05)
            plt.savefig("pngs/post_cov_heat_{}.png".format(params))

        # match DCA with ground-truth
        np.save("npys/ddca_bases_{}.npy".format(params), X_ddca)
        print("Matching {}".format(args.base_encoder_type))
        X_dca_recon, _ = match(X_dca, X_dyn_val[:500], 15000, device)
        # match d-DCA with ground-truth
        print("Matching {}".format(encoder_name))
        X_ddca_recon, _ = match(X_ddca, X_dyn_val[:500], 15000, device)

        # R2 of dca
        r2_dca = 1 - np.sum((X_dca_recon - X_dyn_val[:500]) ** 2) / np.sum(
                (X_dyn_val[:500] - np.mean(X_dyn_val[:500], axis=0)) ** 2)
        print("\nr2_dca:", r2_dca)
        # R2 of ddca
        r2_ddca = 1 - np.sum((X_ddca_recon - X_dyn_val[:500]) ** 2) / np.sum(
                (X_dyn_val[:500] - np.mean(X_dyn_val[:500], axis=0)) ** 2)
        print("r2_ddca:", r2_ddca)
        # store R2's
        r2_vals[snr_idx] = [r2_dca, r2_ddca]
        # store reconstructed signals    
        dca_recons.append(X_dca_recon)
        ddca_recons.append(X_ddca_recon)

    plot_figs(dca_recons, ddca_recons, X_dyn_val[:500], X_clean_val[:500], X_noisy_val[:500], r2_vals, snr_vals, args.base_encoder_type,
              encoder_name, "figs/result_{}.pdf".format(params))


if __name__ == "__main__":
    main(sys.argv[1:])

"""
python3 nonlinear_lorenz.py --encoder_type bgru --dropout 0.5 --ortho_lambda 10.0 --recon_lambda 10.0 --gpuid 0
"""
