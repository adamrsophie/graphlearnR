# Dynamic link prediction on snapshots

Trains a recurrent graph encoder on a
[`snapshot_sequence()`](https://adamrsophie.github.io/graphlearnR/reference/snapshot_sequence.md):
the hidden state after snapshot `k` (computed from snapshots `1, ..., k`
only) scores the links of snapshot `k + 1` against negatives drawn from
non-edges of snapshot `k + 1`. Transitions whose target snapshot index
is in `train_steps` are used for training; the remaining transitions are
evaluated, in time order, with the state carried forward through every
earlier snapshot. No snapshot after `k` ever influences a score for
snapshot `k + 1`.

## Usage

``` r
train_dynamic_link_model(
  snapshots,
  train_steps = NULL,
  hidden_dim = 32L,
  epochs = 50L,
  lr = 0.01,
  neg_ratio = 1,
  seed = NULL,
  device = NULL,
  cell = c("gru", "evolve")
)
```

## Arguments

- snapshots:

  A `snapshot_sequence`.

- train_steps:

  Target snapshot indices used for training (default the first 70
  percent of transitions).

- hidden_dim:

  Hidden size.

- epochs:

  Epochs.

- lr:

  Learning rate.

- neg_ratio:

  Negatives per positive.

- seed:

  Optional integer seed.

- device:

  Device.

- cell:

  `"gru"` for a
  [`graph_gru_cell()`](https://adamrsophie.github.io/graphlearnR/reference/graph_gru_cell.md)
  (node states evolve) or `"evolve"` for an
  [`evolve_gcn_cell()`](https://adamrsophie.github.io/graphlearnR/reference/evolve_gcn_cell.md)
  (GCN weights evolve, EvolveGCN-O).

## Value

An object of class `temporal_fit` with per step test metrics
(`step_metrics`) and averages (`test_metrics`).

## Examples

``` r
if (torch::torch_is_installed()) {
  snaps <- temporal_snapshots(dataset_synthetic_temporal(seed = 1), breaks = 8)
  fit <- train_dynamic_link_model(snaps, hidden_dim = 8, epochs = 3, seed = 1)
  fit
}
#> <temporal_fit> dynamic link prediction, 3 epochs
#>   test:      roc_auc = 0.6566, average_precision = 0.6567 
#>   baseline:  roc_auc = 0.8348, average_precision = 0.8224 
```
