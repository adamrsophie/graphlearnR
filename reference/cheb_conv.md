# Chebyshev spectral graph convolution

The ChebConv layer of Defferrard, Bresson and Vandergheynst (2016):
\$\$X' = \sum\_{k=1}^{K} T\_{k-1}(\hat L) X W_k + b,\$\$ with Chebyshev
polynomials \\T_0(\hat L) X = X\\, \\T_1(\hat L) X = \hat L X\\ and
\\T_k = 2 \hat L T\_{k-1} - T\_{k-2}\\, and the scaled Laplacian \\\hat
L = 2L / \lambda\_{max} - I\\ for the symmetric normalized Laplacian \\L
= I - D^{-1/2} A D^{-1/2}\\. As in PyTorch Geometric's default,
\\\lambda\_{max} = 2\\, so \\\hat L = -D^{-1/2} A D^{-1/2}\\ (self loops
are removed first). The filter reaches `K - 1` hops. Degrees are
weighted in degrees, so the operator is the symmetric Laplacian for
undirected graphs.

## Usage

``` r
cheb_conv(in_features, out_features, K = 3L, bias = TRUE)
```

## Arguments

- in_features, out_features:

  Feature sizes.

- K:

  Number of Chebyshev terms (filter order plus one).

- bias:

  Learn an additive bias.

## Value

An `nn_module` called as `layer(x, edge_index, edge_weight = NULL)`.

## Details

Input `x` is N by F; output N by F'. Edge input: `edge_weight` only.
Cost: \\K - 1\\ sparse propagations, \\O(K E F)\\, plus \\O(K N F F')\\
for the transforms; memory holds the K terms \\T_k X\\ (\\K N F\\).

## References

Defferrard, Bresson and Vandergheynst (2016) Convolutional neural
networks on graphs with fast localized spectral filtering.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- dataset_karate()
  tt <- graph_tensors(g)
  cheb_conv(34, 8, K = 3)(tt$x, tt$edge_index)$shape
}
#> [1] 34  8
```
