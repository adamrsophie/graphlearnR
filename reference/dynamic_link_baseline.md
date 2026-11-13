# Heuristic baseline for dynamic link prediction

Scores the links of snapshot `k` with a heuristic computed from
snapshots `1, ..., k - 1` only: the number of earlier interactions
between the pair (`"history"`) or the product of earlier degrees
(`"preferential"`), with the same negative sampling as
[`train_dynamic_link_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_dynamic_link_model.md).

## Usage

``` r
dynamic_link_baseline(
  snapshots,
  steps,
  neg_ratio = 1,
  seed = NULL,
  method = c("history", "preferential")
)
```

## Arguments

- snapshots:

  A `snapshot_sequence`.

- steps:

  Target snapshot indices.

- neg_ratio:

  Negatives per positive.

- seed:

  Optional integer seed.

- method:

  `"history"` or `"preferential"`.

## Value

Named vector of mean ROC-AUC and average precision.

## Examples

``` r
snaps <- temporal_snapshots(dataset_synthetic_temporal(seed = 1), breaks = 8)
dynamic_link_baseline(snaps, steps = 7:8, seed = 1)
#>           roc_auc average_precision 
#>         0.8330262         0.8213880 
```
