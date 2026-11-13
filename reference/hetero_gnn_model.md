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
  conv = c("sage", "graph_conv", "gat", "rgcn", "hgt", "han"),
  heads = 2L,
  dropout = 0.5,
  aggr = "mean",
  relation_aggr = "sum",
  num_bases = NULL,
  temporal_encoding = FALSE
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

  Attention heads for `"gat"`, `"han"` and `"hgt"`.

- dropout:

  Dropout probability.

- aggr:

  Neighbor aggregation inside each relation.

- relation_aggr:

  Combination across relations, `"sum"` or `"mean"`.

- num_bases:

  Basis count for `"rgcn"`.

- temporal_encoding:

  For `"hgt"`, use the relative temporal encoding with the relations'
  `edge_time`.

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

- `conv = "han"`: HAN (Wang et al. 2019): node level attention per
  relation (`gat_conv`) and semantic attention across the relations
  arriving at a type (`hetero_conv(aggr = "semantic")`). HAN uses
  metapath relations; add them with
  [`add_metapath()`](https://adamrsophie.github.io/graphlearnR/reference/add_metapath.md)
  (all relations of `hg` are used, so the plain relations take part
  too).

- `conv = "hgt"`: heterogeneous graph transformer layers, see
  [`hgt_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hgt_conv.md);
  attention is normalized over all incoming relations of a node.

Embedding tables make the model transductive for featureless types: it
can only predict for nodes seen during training.

## Examples

``` r
if (torch::torch_is_installed()) {
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
  m <- hetero_gnn_model(hg, hidden_dim = 16, output_dim = 3, target_type = "drug")
  m
}
#> An `nn_module` containing 4,467 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • input: <nn_module> #192 parameters
#> • layers: <nn_module_list> #4,224 parameters
#> • head: <nn_linear> #51 parameters
```
