# Heterogeneous graph tensors

Converts every node type and relation of a heterogeneous graph to
`torch` tensors, once, on one device.

## Usage

``` r
hetero_tensors(hg, device = NULL)
```

## Arguments

- hg:

  A `hetero_graph_data` object.

- device:

  Device, see
  [`graph_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/graph_tensors.md).

## Value

A list with named lists `x`, `y`, `edge_index`, `edge_attr`,
`edge_weight` and the masks, plus `num_nodes`, `node_types` and
`relations` (the result of
[`edge_types()`](https://adamrsophie.github.io/graphlearnR/reference/edge_types.md)).

## Examples

``` r
if (torch::torch_is_installed()) {
  ht <- hetero_tensors(dataset_synthetic_hetero(seed = 1))
  names(ht)
}
#>  [1] "x"           "y"           "train_mask"  "val_mask"    "test_mask"  
#>  [6] "edge_index"  "edge_attr"   "edge_weight" "edge_time"   "node_index" 
#> [11] "num_nodes"   "node_types"  "relations"   "device"     
```
