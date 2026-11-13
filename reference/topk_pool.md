# Top-k pooling (hierarchical, experimental)

Graph U-Net style pooling (Gao and Ji 2019): nodes are scored by \\y = X
p / \lVert p \rVert\\, the top `ceiling(ratio * n_g)` nodes of each
graph are kept, their features are scaled by \\\tanh(y)\\ and the edges
between kept nodes are retained.

## Usage

``` r
topk_pool(in_features, ratio = 0.5)

sag_pool(in_features, ratio = 0.5, gnn = "graph_conv")
```

## Arguments

- in_features:

  Node feature size.

- ratio:

  Fraction of nodes kept per graph.

- gnn:

  For `sag_pool()`, the layer that scores nodes: `"graph_conv"`, `"gcn"`
  or `"sage"`, or a module mapping `(x, edge_index)` to one score per
  node.

## Value

An `nn_module` returning a list with `x`, `edge_index`, `batch`, `perm`
(kept node indices in the input) and `score`.

## Details

`sag_pool()` is self attention graph pooling (SAGPool, Lee, Lee and Kang
2019): the scores come from a graph convolution, \\y = \mathrm{GNN}(X,
A)\\, so they depend on the neighborhood; the rest is as in
`topk_pool()`.
