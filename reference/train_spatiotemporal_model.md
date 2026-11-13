# Train a spatiotemporal forecaster

Fits a
[`temporal_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_gnn_model.md)
to predict the target at step `s + horizon` from features at steps
`s - lags + 1, ..., s`. Windows are split in time order: the first
`train` fraction of target steps trains the model, the next `val`
fraction validates, and the rest is the test period. Windows at the end
of the training and validation periods whose target is observed after
the first forecast origin of the next period are dropped (a purge of
`horizon - 1` windows), so no training target is later than a validation
or test forecast origin.

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

## Examples

``` r
if (torch::torch_is_installed()) {
  sig <- dataset_synthetic_signal(seed = 1)
  fit <- train_spatiotemporal_model(sig, lags = 3, hidden_dim = 8, epochs = 3, seed = 1)
  fit
}
#> <temporal_fit> spatiotemporal forecasting, 3 epochs
#>   test:      mse = 0.01026, mae = 0.08195 
#>   baseline:  mse = 0.01409, mae = 0.09513 
```
