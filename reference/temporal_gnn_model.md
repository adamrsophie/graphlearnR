# Recurrent temporal GNN

A
[`graph_gru_cell()`](https://adamrsophie.github.io/graphlearnR/reference/graph_gru_cell.md)
followed by a linear readout per node and step.
`forward(x_seq, edge_index_seq, edge_weight_seq)` takes lists of per
step tensors (one graph per step, or the same graph repeated) and
returns a list with per step outputs and the final hidden state. Outputs
at step `s` depend only on inputs at steps `1, ..., s`.

## Usage

``` r
temporal_gnn_model(input_dim, hidden_dim = 32L, output_dim = 1L)
```

## Arguments

- input_dim:

  Node feature size.

- hidden_dim:

  Hidden size.

- output_dim:

  Output size (per node), or `NULL` to return hidden states.

## Value

An `nn_module` of class `temporal_gnn`.

## Examples

``` r
if (torch::torch_is_installed()) {
  m <- temporal_gnn_model(input_dim = 2, hidden_dim = 8, output_dim = 1)
  m
}
#> An `nn_module` containing 297 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • cell: <graph_gru_cell> #288 parameters
#> • head: <nn_linear> #9 parameters
```
