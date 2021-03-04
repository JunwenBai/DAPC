# Deep Autoencoding Predictive Components

### Overview

<div align=center><img src="figs/DAPC.png" width="70%"></div>

Deep Autoencoding Predictive Components (DAPC) is a self-supervised representation learning method for sequence data, based 
on the intuition that useful representations of sequence data should exhibit a simple structure in the latent space. 

We encourage this latent structure by maximizing an estimate of `predictive information` (PI) of latent feature sequences, and 
regularize the learning through masked reconstruction; the full learning objective is described in [[1]](https://arxiv.org/abs/2010.03135). Here we use the same estimate of predictive information from the 
recent work `Dynamical Components Analysis` [[3]](https://github.com/BouchardLab/DynamicalComponentsAnalysis) (and our implementation 
of PI is modified from theirs). The masked reconstruction loss was applied to pretraining encoders for speech recognition 
in [[2]](https://arxiv.org/abs/2001.10603).

This repository mainly demonstrates the Lorenz Attractor experiments.

<p float="middle">
  <img src="figs/raw.png" width="23%" />
  <img src="figs/30d.png" width="23%" /> 
  <img src="figs/30d_noisy.png" width="23%" />
  <img src="figs/recovered.png" width="23%" />
</p>

Leftmost:  ground-truth 3d signals. Middle left: lifted 30d signals. Middle right: noisy lifted 30d signals. Rightmost: unsupervised recovery of the 3d signals by DAPC.

### Requirements

- Python 3.7+
- numpy 1.17.3
- matplotlib
- PyTorch 1.5.0

Older versions might work as well.

### Usage

Download the repo

```
git clone https://github.com/JunwenBai/DAPC.git
```

To run the deterministic DAPC

```
./run_ddapc.sh
```

To run the probabilistic DAPC

```
./run_vdapc.sh
```

One can inspect the bashes to see all the options for training. By default, we use `gpu:0`.

### Paper

If you are interested in our work, please consider cite the following paper:

```bibtex
@inproceedings{
	bai2021representation,
	title={Representation Learning for Sequence Data with Deep Autoencoding Predictive Components},
	author={Junwen Bai and Weiran Wang and Yingbo Zhou and Caiming Xiong},
	booktitle={International Conference on Learning Representations},
	year={2021},
	url={https://openreview.net/forum?id=Naqw7EHIfrv}
}
```

### References

[1] Junwen Bai, Weiran Wang, Yingbo Zhou, and Caiming Xiong. Representation Learning for Sequence Data with Deep Autoencoding Predictive Components. In *International Conference on Learning Representations*, 2021.

[2] Weiran Wang, Qingming Tang, and Karen Livescu. Unsupervised Pre-training of Bidirectional Speech Encoders via Masked Reconstruction. In *ICASSP*, 2020.

[3] Clark, D., Livezey, J. and Bouchard, K.. Unsupervised discovery of temporal structure in noisy data with dynamical components analysis. In *Advances in Neural Information Processing Systems*, 2019.
