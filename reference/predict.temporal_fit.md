# Predict with a temporal model

Continues a fitted temporal model on new inputs. Every method uses only
information available at the prediction time.

## Usage

``` r
# S3 method for class 'temporal_fit'
predict(
  object,
  new_data = NULL,
  pairs = NULL,
  time = NULL,
  step = NULL,
  ends = NULL,
  type = c("prob", "score"),
  ...
)
```

## Arguments

- object:

  A `temporal_fit`.

- new_data:

  New events (TGN), a `snapshot_sequence` (dynamic links) or a
  `spatiotemporal_signal` (forecasting).

- pairs:

  2 by K matrix of node pairs to score (link models).

- time:

  Prediction time, one value or one per pair (TGN).

- step:

  Target snapshot index (dynamic links).

- ends:

  Forecast origins, time step indices (forecasting).

- type:

  `"prob"` or `"score"` (logit) for link models.

- ...:

  Unused.

## Value

Link models: a numeric vector with one entry per pair. Forecasting: a
matrix with one row per origin and one column per node.

## Details

- **TGN**
  ([`train_tgn()`](https://adamrsophie.github.io/graphlearnR/reference/tgn_model.md)):
  scores candidate pairs `pairs` at times `time`. The fit keeps the
  memory, last update times and neighbor store after all training,
  validation and test events. Events observed since then can be given as
  `new_data` (a `temporal_graph_data` with times at or after the last
  event of the fit); they update the state first, in the same batches as
  during training, and no score uses an event at or after its own time.
  Prediction times must not precede the last observed event.

- **Dynamic link prediction**
  ([`train_dynamic_link_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_dynamic_link_model.md)):
  scores `pairs` as links of snapshot `step` of `new_data` (a
  `snapshot_sequence`), from the recurrent state built on snapshots
  `1, ..., step - 1` only.

- **Spatiotemporal forecasting**
  ([`train_spatiotemporal_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_spatiotemporal_model.md)):
  forecasts the targets `horizon` steps after each forecast origin in
  `ends`, from the `lags` feature steps ending at that origin of
  `new_data` (a `spatiotemporal_signal`).

## Examples

``` r
if (torch::torch_is_installed()) {
  sig <- dataset_synthetic_signal(seed = 1)
  fit <- train_spatiotemporal_model(sig, lags = 3, hidden_dim = 8, epochs = 2, seed = 1)
  dim(predict(fit, sig, ends = 10:12))
}
#> [1]  3 20
```
