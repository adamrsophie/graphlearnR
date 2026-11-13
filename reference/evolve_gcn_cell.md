# Evolving GCN cell (EvolveGCN-O)

The EvolveGCN-O cell of Pareja et al. (2020): the GCN weight matrix, not
a node state, is the recurrent state. At each snapshot \$\$W_t =
\mathrm{GRU}(W\_{t-1}, W\_{t-1}),\quad H_t = \mathrm{ReLU}(\hat A_t X_t
W_t),\$\$ with the rows of \\W\\ as the GRU batch and \\\hat A_t\\ the
GCN propagation matrix of snapshot \\t\\ (self loops, symmetric
normalization). Because only weights evolve, nodes that appear late are
handled like any other node.

## Usage

``` r
evolve_gcn_cell(in_features, hidden_features)
```

## Arguments

- in_features:

  Input feature size.

- hidden_features:

  Output size.

## Value

An `nn_module` called as `cell(x, edge_index, h, edge_weight)`.

## Details

The call signature matches
[`graph_gru_cell()`](https://adamrsophie.github.io/graphlearnR/reference/graph_gru_cell.md)
so the cell can replace it in
[`train_dynamic_link_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_dynamic_link_model.md).
The evolving weight is kept inside the module between calls: a call with
`h = NULL` restarts from the learned initial weight \\W_0\\; any other
`h` (the previous output) means "continue", and its value is not used.

## References

Pareja et al. (2020) EvolveGCN: evolving graph convolutional networks
for dynamic graphs. AAAI.

## Examples

``` r
if (torch::torch_is_installed()) {
  cell <- evolve_gcn_cell(3, 8)
  ei <- graph_tensors(dataset_karate())$edge_index
  x <- torch::torch_randn(34, 3)
  h1 <- cell(x, ei)
  h2 <- cell(x, ei, h1)
  h2$shape
}
#> [1] 34  8
```
