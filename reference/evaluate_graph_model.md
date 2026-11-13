# Evaluate a trained graph model

Computes task metrics on a held out set, using
[`evaluate_predictions()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_predictions.md).

## Usage

``` r
evaluate_graph_model(
  fit,
  new_data = NULL,
  split = c("test", "val", "train"),
  ks = c(1L, 10L, 50L),
  filtered_negatives = NULL,
  seed = NULL
)
```

## Arguments

- fit:

  A `graph_fit`.

- new_data:

  Evaluation data; defaults to the training data object.

- split:

  `"test"`, `"val"` or `"train"`.

- ks:

  Cut-offs for Hits@K.

- filtered_negatives:

  For homogeneous link prediction: number of filtered corrupted tails
  per positive for MRR and Hits@K, or `NULL`.

- seed:

  Optional seed for drawing the corrupted tails.

## Value

A data frame with `metric` and `value`.

## Details

- Node tasks: nodes in `new_data`'s `split` mask (`"test"` by default).

- Graph tasks: every graph in `new_data`.

- Link prediction: `split$test_pos` against `split$test_neg` (or the
  validation pairs with `split = "val"`), with message passing on the
  training graph only. ROC-AUC and average precision use those pairs; by
  default MRR and Hits@K rank every positive against the shared
  negatives. With `filtered_negatives = k`, each positive \\(u, v)\\ is
  instead ranked against `k` corrupted tails \\(u, v')\\ drawn by
  [`corrupt_tails()`](https://adamrsophie.github.io/graphlearnR/reference/corrupt_tails.md)
  so that no \\(u, v')\\ is a known positive pair of the split (the
  filtered setting of Bordes et al. 2013).

- Edge tasks: edges in the `split` edge mask.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification",
                           epochs = 20, seed = 1)
  evaluate_graph_model(fit)
}
#>              metric     value
#> 1          accuracy 0.8333333
#> 2 balanced_accuracy 0.8333333
#> 3   precision_macro 0.8750000
#> 4      recall_macro 0.8333333
#> 5          f1_macro 0.8285714
#> 6               ece 0.4036021
#> 7             brier 0.3302357
#> 8           roc_auc 1.0000000
#> 9 average_precision 1.0000000
```
