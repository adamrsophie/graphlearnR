# Global attention pooling

Soft attention readout (Li et al. 2016): each node gets a gate score
\\s_i = \mathrm{gate}(x_i)\\, scores are normalized with a softmax
within each graph, and the graph vector is \\\sum_i
\mathrm{softmax}(s)\_i\\ \mathrm{nn}(x_i)\\. Permutation invariant.

## Usage

``` r
global_attention_pool(in_features, gate_nn = NULL, nn = NULL)
```

## Arguments

- in_features:

  Node feature size.

- gate_nn:

  Module mapping features to one score; default a linear layer.

- nn:

  Optional module applied to node features before weighting.

## Value

An `nn_module` called as `module(x, batch, num_graphs)`.

## Examples

``` r
if (torch::torch_is_installed()) {
  pool <- global_attention_pool(4)
  x <- torch::torch_randn(5, 4)
  pool(x, batch = c(1, 1, 2, 2, 2), num_graphs = 2)$shape
}
#> [1] 2 4
```
