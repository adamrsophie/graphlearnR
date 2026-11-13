# Train a spatiotemporal forecaster

Fits a
[`temporal_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_gnn_model.md)
to predict the target at step `s + horizon` from features at steps
`s - lags + 1, ..., s`. Windows are split in time order: the first
`train` fraction of target steps trains the model, the next `val`
fraction validates, and the rest is the test period. No window used for
training has a target time inside the validation or test period.

## Usage

``` r
train_spatiotemporal_model(
  signal,
  lags = 4L,
  horizon = 1L,
  hidden_dim = 32L,
  epochs = 50L,
  lr = 0.01,
  train = 0.7,
  val = 0.15,
  seed = NULL,
  device = NULL
)
```

## Arguments

- signal:

  A
  [`spatiotemporal_signal()`](https://adamrsophie.github.io/graphlearnR/reference/spatiotemporal_signal.md).

- lags:

  Input window length.

- horizon:

  Steps ahead.

- hidden_dim:

  Hidden size.

- epochs:

  Epochs.

- lr:

  Learning rate.

- train, val:

  Fractions of windows for training and validation.

- seed:

  Optional integer seed.

- device:

  Device.

## Value

An object of class `temporal_fit` with the model, history, window split
and test metrics (`test_metrics`), alongside a naive last value baseline
(`baseline_metrics`).
