# Evaluate node or graph embeddings

Two task independent checks of a learned representation, for example
`predict(fit, type = "embedding")`:

## Usage

``` r
evaluate_embeddings(
  embedding,
  labels,
  train,
  lambda = 0.001,
  k = NULL,
  seed = NULL
)
```

## Arguments

- embedding:

  Numeric matrix, one row per node or graph.

- labels:

  Factor (classification) or numeric vector, one per row.

- train:

  Logical vector of rows used to fit the probe.

- lambda:

  Ridge penalty.

- k:

  Number of clusters; `NULL` uses the number of classes. Ignored (no
  clustering) for numeric labels unless given.

- seed:

  Optional seed for k-means starts.

## Value

A data frame with columns `metric` and `value`.

## Details

- **Downstream prediction (linear probe).** A ridge regression from the
  standardized embedding to one hot labels (classification) or to the
  numeric target is fitted on the `train` rows and scored on the others.
  Classification reports accuracy and macro F1; regression reports RMSE
  and \\R^2\\. A linear probe measures how much label information is
  linearly available, not the best attainable model.

- **Clustering.** k-means with `k` centers (default: the number of
  classes) on the standardized embedding, compared with the labels by
  normalized mutual information and the adjusted Rand index.

The split of `train` rows must follow the dependence structure of the
data (for example
[`split_nodes()`](https://adamrsophie.github.io/graphlearnR/reference/split_nodes.md)
masks), not a random mix of dependent observations.

## Examples

``` r
g <- dataset_sbm(c(30, 30), p_in = 0.3, p_out = 0.02, seed = 1)
emb <- laplacian_pe(g, 4)
tr <- rep(c(TRUE, FALSE), 30)
evaluate_embeddings(emb, g$y, train = tr, seed = 1)
#>           metric         value
#> 1 probe_accuracy  1.0000000000
#> 2 probe_f1_macro  1.0000000000
#> 3    cluster_nmi  0.0008126869
#> 4    cluster_ari -0.0159359497
```
