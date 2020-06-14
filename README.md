# Deep Dynamical Component Analysis

This code implements and analyses deep DCA.

### Requirements

- Python 3.7+
- numpy 1.17.3
- matplotlib
- PyTorch 1.5.0

Older versions might work as well.

### Usage

After downloading the code repo, one should create two folders `results` and `figs` to store the data and figures respectively.

Then simply run

```python nonlinear_lorenz.py 4 0```

`4` the the size of the time window `T`. `0` is the random seed.

### References

Clark, David, Jesse Livezey, and Kristofer Bouchard. "Unsupervised Discovery of Temporal Structure in Noisy Data with Dynamical Components Analysis." Advances in Neural Information Processing Systems. 2019.

[DCA](https://github.com/BouchardLab/DynamicalComponentsAnalysis)
