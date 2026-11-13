# Heterogeneous convolution wrapper

Applies one bipartite layer per relation and sums (or averages) the
results arriving at each target node type, the scheme of PyG's
`HeteroConv`. Node types that receive no relation keep their input
features.

## Usage

``` r
hetero_conv(convs, aggr = c("sum", "mean"))
```

## Arguments

- convs:

  Named list of layers keyed by relation key; each is called as
  `layer(list(x_src, x_dst), edge_index)`.

- aggr:

  `"sum"` or `"mean"` over relations.

## Value

An `nn_module` called as `module(x_dict, edge_index_dict)`.

## Examples

``` r
if (torch::torch_is_installed()) {
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
  convs <- lapply(stats::setNames(nm = edge_types(hg)), function(r) sage_conv(16, 8))
  conv <- hetero_conv(convs)
  conv
}
#> An `nn_module` containing 1,584 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • convs: <nn_module> #1,584 parameters
```
