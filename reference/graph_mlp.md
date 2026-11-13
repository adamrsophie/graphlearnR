# Multilayer perceptron

A plain MLP used inside
[`gin_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
and as prediction heads.

## Usage

``` r
graph_mlp(dims, activation = nn_relu, dropout = 0, batch_norm = FALSE)
```

## Arguments

- dims:

  Integer vector of layer sizes, including input and output.

- activation:

  Activation module constructor between layers.

- dropout:

  Dropout probability between layers.

- batch_norm:

  Insert batch normalization between layers.

## Value

An `nn_module`.

## Examples

``` r
if (torch::torch_is_installed()) {
  m <- graph_mlp(c(4, 8, 2))
  m(torch::torch_randn(3, 4))$shape
}
#> [1] 3 2
```
