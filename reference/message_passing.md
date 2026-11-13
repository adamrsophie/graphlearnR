# Generic message passing

Computes one round of \$\$h_i' = U\left(h_i, \bigoplus\_{j \in N(i)}
M(h_i, h_j, e\_{ji})\right)\$\$ for every target node `i`, where `N(i)`
holds the sources of edges that point to `i`. The message function sees
whole edge tensors, so the work is vectorized over edges; aggregation
uses the
[scatter](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
kernels.

## Usage

``` r
message_passing(
  x,
  edge_index,
  message = NULL,
  aggr = "sum",
  update = NULL,
  edge_attr = NULL,
  edge_weight = NULL,
  num_nodes = NULL
)
```

## Arguments

- x:

  Node feature tensor, or a list `list(x_src, x_dst)`.

- edge_index:

  Long tensor, 2 by E, one based.

- message:

  Message function; the default sends `x_j` (scaled by `edge_weight`
  when given).

- aggr:

  Aggregation name or names, see
  [`scatter_aggregate()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md),
  or a function `function(messages, index, dim_size)`.

- update:

  Optional update function.

- edge_attr:

  Optional edge feature tensor (E rows).

- edge_weight:

  Optional edge weight tensor (length E).

- num_nodes:

  Number of target nodes. Defaults to the rows of the target feature
  tensor.

## Value

A tensor with one row per target node.

## Details

The message function receives named arguments `x_i` (target features
gathered per edge), `x_j` (source features per edge), `edge_attr`,
`edge_weight`, `index` (target index per edge) and `num_nodes`, and must
return one row per edge. Unused arguments can be absorbed with `...`.
The update function receives `aggr_out` and `x` (target node features)
and returns the new node features.

For bipartite or heterogeneous relations pass `x = list(x_src, x_dst)`;
`edge_index[1, ]` then indexes rows of `x_src` and `edge_index[2, ]`
rows of `x_dst`.

Permutation equivariance holds whenever `message` and `update` treat
edges and nodes row wise, which is the case for all built in layers.

## Examples

``` r
if (torch::torch_is_installed()) {
  x <- torch::torch_tensor(matrix(1:6, 3, 2), dtype = torch::torch_float())
  ei <- torch::torch_tensor(rbind(c(1L, 2L), c(3L, 3L)),
                            dtype = torch::torch_long())
  message_passing(x, ei, aggr = "sum")
}
#> torch_tensor
#>  0  0
#>  0  0
#>  3  9
#> [ CPUFloatType{3,2} ]
```
