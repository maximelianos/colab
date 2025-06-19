# Neural Times Series

Intrinsic Dimensionality in Neural Times Series Data.

Sources are in these notebooks:
1. data-explore.ipynb
2. neuron-activity.ipynb
3. lstm.ipynb

Place data in `./data`

## LSTM method

The autoencoder model takes 1-D time series for one neuron, compresses it to a latent vector
and reconstructs the original time series. The Intrinsic Dimensionality for the set of latent vectors
is computed.

For faster performance, disk caching is used for:
* gaussian smoothing of original signal and subsampling for the whole dataset
* the mean and standard deviation of signal for each session
* predicted time series
