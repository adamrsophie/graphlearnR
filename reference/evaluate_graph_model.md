# Evaluate a trained graph model

Computes task metrics on a held out set, using
[`evaluate_predictions()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_predictions.md).

## Usage

``` r
evaluate_graph_model(
  fit,
  new_data = NULL,
  split = c("test", "val", "train"),
  ks = c(1L, 10L, 50L)
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

## Value

A data frame with `metric` and `value`.

## Details

- Node tasks: nodes in `new_data`'s `split` mask (`"test"` by default).

- Graph tasks: every graph in `new_data`.

- Link prediction: `split$test_pos` against `split$test_neg` (or the
  validation pairs with `split = "val"`), with message passing on the
  training graph only.

- Edge tasks: edges in the `split` edge mask.
