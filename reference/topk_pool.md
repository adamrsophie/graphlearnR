# Top-k pooling (hierarchical, experimental)

Graph U-Net style pooling (Gao and Ji 2019): nodes are scored by \\y = X
p / \lVert p \rVert\\, the top `ceiling(ratio * n_g)` nodes of each
graph are kept, their features are scaled by \\\tanh(y)\\ and the edges
between kept nodes are retained.

## Usage

``` r
topk_pool(in_features, ratio = 0.5)
```

## Arguments

- in_features:

  Node feature size.

- ratio:

  Fraction of nodes kept per graph.

## Value

An `nn_module` returning a list with `x`, `edge_index`, `batch`, `perm`
(kept node indices in the input) and `score`.
