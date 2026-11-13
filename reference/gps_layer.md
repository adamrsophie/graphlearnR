# GPS layer: local message passing plus global attention

One layer of the general, powerful, scalable graph transformer (GPS,
Rampasek et al. 2022), as PyTorch Geometric's `GPSConv`: \$\$h_L =
\mathrm{LN}(x + \mathrm{conv}(x)),\quad h_G = \mathrm{LN}(x +
\mathrm{MHA}(x)),\quad h = h_L + h_G,\quad \mathrm{out} =
\mathrm{LN}(h + \mathrm{MLP}(h)).\$\$ The local part is any message
passing layer whose input and output size is `dim` (for example
[`gin_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
or
[`gat_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
with `concat = FALSE`); the global part is multi head self attention
among the nodes of each graph of the batch (padded with
[`to_dense_batch()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_batch.md)
and masked, so graphs never attend to each other). Layer norm is used
where PyG defaults to batch norm. Positional encodings
([`laplacian_pe()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md),
[`random_walk_pe()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md))
are added to the input features by the caller, as in the paper.

## Usage

``` r
gps_layer(dim, conv = NULL, heads = 4L, dropout = 0, attn_dropout = 0)
```

## Arguments

- dim:

  Feature size.

- conv:

  Local message passing module mapping `dim` to `dim`, or `NULL` for
  attention only.

- heads:

  Attention heads (`dim` must be divisible by `heads`).

- dropout:

  Dropout after each part.

- attn_dropout:

  Dropout on attention weights.

## Value

An `nn_module` called as `layer(x, edge_index, batch = NULL, ...)`;
`...` (for example `edge_attr`) goes to `conv`.

## Details

Both parts are permutation equivariant. Memory of the global part is
\\O(B L^2 H)\\ for `B` graphs of at most `L` nodes; linear attention
variants (Performer) are not provided.

## References

Rampasek et al. (2022) Recipe for a general, powerful, scalable graph
transformer. NeurIPS.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- dataset_karate()
  tt <- graph_tensors(g)
  layer <- gps_layer(8, conv = gin_conv(graph_mlp(c(8, 8, 8))), heads = 2)
  layer(torch::torch_randn(34, 8), tt$edge_index)$shape
}
#> [1] 34  8
```
