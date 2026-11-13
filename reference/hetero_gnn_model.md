# Heterogeneous graph neural network

A per type input layer maps every node type to `hidden_dim` (a linear
layer for types with features, a learned embedding table for types
without), followed by `num_layers` relation aware layers and an optional
head for `target_type`.

## Usage

``` r
hetero_gnn_model(
  hg,
  hidden_dim = 64L,
  output_dim = NULL,
  target_type = NULL,
  num_layers = 2L,
  conv = c("sage", "graph_conv", "gat", "rgcn"),
  heads = 2L,
  dropout = 0.5,
  aggr = "mean",
  relation_aggr = "sum",
  num_bases = NULL
)
```

## Arguments

- hg:

  A `hetero_graph_data` object used for its schema (types, relations,
  feature sizes, node counts); its data are not stored.

- hidden_dim:

  Hidden size.

- output_dim:

  Output size of the head, or `NULL` for embeddings only.

- target_type:

  Node type that gets the head.

- num_layers:

  Number of relation aware layers.

- conv:

  Layer type.

- heads:

  Attention heads for `"gat"`.

- dropout:

  Dropout probability.

- aggr:

  Neighbor aggregation inside each relation.

- relation_aggr:

  Combination across relations, `"sum"` or `"mean"`.

- num_bases:

  Basis count for `"rgcn"`.

## Value

An `nn_module` of class `hetero_gnn`, called as `model(ht)` with `ht`
from
[`hetero_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_tensors.md);
it returns a named list of per type outputs.

## Details

- `conv = "rgcn"`: relational GCN (Schlichtkrull et al. 2018) on the
  homogeneous view; each relation has its own weight, neighbors are
  averaged per relation.

- `conv = "sage"`, `"graph_conv"`, `"gat"`: one bipartite layer per
  relation, combined with
  [`hetero_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_conv.md).
  `"gat"` gives relation specific attention (a heterogeneous GAT in the
  sense of PyG's `to_hetero` applied to `GATConv`; it is not HAN or
  HGT).

Embedding tables make the model transductive for featureless types: it
can only predict for nodes seen during training.
