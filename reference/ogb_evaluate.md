# Official OGB evaluation metrics

Computes the metric that the Open Graph Benchmark uses for a dataset,
following the logic of OGB's `Evaluator`:

## Usage

``` r
ogb_evaluate(name, truth, pred)
```

## Arguments

- name:

  OGB dataset name.

- truth:

  Labels: a vector or a matrix with one column per task (`NA` for
  missing labels).

- pred:

  Predicted scores (any monotone score for the molecular datasets, for
  example logits) with the same shape as `truth`, or predicted classes
  for the node datasets.

## Value

A named numeric scalar.

## Details

- `ogbg-molhiv`: ROC-AUC; `ogbg-molpcba`: average precision. For both,
  the metric is computed per task over the rows whose label is observed
  (not `NA`), tasks without both classes are skipped, and the mean over
  tasks is returned.

- `ogbn-arxiv`, `ogbn-products`: accuracy.

Results are comparable to the leaderboards only with the official split
(the split stored by
[`read_ogb_node()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md)
and
[`read_ogb_graph()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md)).

## Examples

``` r
ogb_evaluate("ogbg-molhiv", c(0, 1, 1, 0), c(0.1, 0.8, 0.6, 0.3))
#> rocauc 
#>      1 
ogb_evaluate("ogbg-molpcba", cbind(c(0, 1, NA, 1), c(1, 0, 0, NA)),
             cbind(c(0.2, 0.9, 0.5, 0.7), c(0.6, 0.1, 0.3, 0.2)))
#> ap 
#>  1 
```
