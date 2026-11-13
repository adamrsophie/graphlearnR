# Graph transformer layer with structural attention bias

Full multi head self attention among the nodes of each graph (never
across graphs of a batch), with a learned scalar bias per head for every
shortest path distance, as in Graphormer (Ying et al. 2021). Uses the
pre-layer-norm arrangement: \\h = h + \mathrm{MHA}(\mathrm{LN}(h))\\,
then \\h = h + \mathrm{FFN}(\mathrm{LN}(h))\\.

## Usage

``` r
graph_transformer_layer(dim, heads = 4L, max_dist = 5L, dropout = 0.1)
```

## Arguments

- dim:

  Feature size.

- heads:

  Attention heads (`dim` must be divisible by `heads`).

- max_dist:

  Largest distinguished distance.

- dropout:

  Dropout probability.

## Value

An `nn_module` called as `layer(x, batch, dist)` where `dist` is a `B`
by `L` by `L` long tensor of clipped distances (padding uses
`max_dist + 2`).

## Details

Graphormer's edge encoding along shortest paths is not implemented.
Memory is \\O(B L^2 H)\\ for `B` graphs of at most `L` nodes and `H`
heads, so this layer suits small graphs such as molecules; for large
graphs use the sparse
[`transformer_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md).

The layer is permutation equivariant: distances and attention are
functions of node pairs, not of node order.

## Examples

``` r
if (torch::torch_is_installed()) {
  layer <- graph_transformer_layer(8, heads = 2)
  layer
}
#> An `nn_module` containing 616 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • ln1: <nn_layer_norm> #16 parameters
#> • ln2: <nn_layer_norm> #16 parameters
#> • qkv: <nn_linear> #216 parameters
#> • proj: <nn_linear> #72 parameters
#> • ffn: <nn_sequential> #280 parameters
#> • dist_bias: <nn_embedding> #16 parameters
```
