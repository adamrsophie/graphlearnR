# Recurrent graph convolution cell

A GRU whose input and hidden transforms are graph convolutions: \$\$z =
\sigma(G\_{xz}(x) + G\_{hz}(h)),\quad r = \sigma(G\_{xr}(x) +
G\_{hr}(h))\$\$ \$\$\tilde h = \tanh(G\_{xh}(x) + G\_{hh}(r \odot
h)),\quad h' = z \odot h + (1 - z) \odot \tilde h\$\$ with \\G\\ a
[`gcn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md).
This follows GConvGRU (Seo et al. 2018) with the Chebyshev convolution
replaced by a first order GCN.

## Usage

``` r
graph_gru_cell(in_features, hidden_features)
```

## Arguments

- in_features:

  Input feature size.

- hidden_features:

  Hidden state size.

## Value

An `nn_module` called as `cell(x, edge_index, h, edge_weight)`.

## Examples

``` r
if (torch::torch_is_installed()) {
  cell <- graph_gru_cell(3, 8)
  g <- dataset_karate()
  x <- torch::torch_randn(34, 3)
  ei <- graph_tensors(g)$edge_index
  cell(x, ei)$shape
}
#> [1] 34  8
```
