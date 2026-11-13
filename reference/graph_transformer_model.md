# Graphormer style graph transformer

Input projection plus centrality encoding (a learned embedding of each
node's clipped degree, added to the features), `num_layers`
[`graph_transformer_layer()`](https://adamrsophie.github.io/graphlearnR/reference/graph_transformer_layer.md)s
with shortest path attention bias, a final layer norm, and either a node
level linear head or global pooling and an MLP head. Shortest path
distances are computed in R for each batch, which costs \\O(\sum_g n_g
(n_g + e_g))\\ per forward pass.

## Usage

``` r
graph_transformer_model(
  input_dim,
  hidden_dim = 64L,
  output_dim = 1L,
  num_layers = 2L,
  heads = 4L,
  max_dist = 5L,
  max_degree = 10L,
  dropout = 0.1,
  pool = "mean"
)
```

## Arguments

- input_dim:

  Node feature size.

- hidden_dim:

  Model width.

- output_dim:

  Output size.

- num_layers:

  Transformer layers.

- heads:

  Attention heads.

- max_dist:

  Largest distinguished shortest path distance.

- max_degree:

  Degrees above this share one embedding.

- dropout:

  Dropout probability.

- pool:

  `NULL` for node level output, or `"mean"`, `"sum"`, `"max"`.

## Value

An `nn_module` of class `graph_transformer`.

## Details

The forward signature matches
[`gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/gnn_model.md),
so the model works with
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md).

## References

Ying et al. (2021) Do transformers really perform badly for graph
representation? NeurIPS.

## Examples

``` r
if (torch::torch_is_installed()) {
  m <- graph_transformer_model(input_dim = 34, hidden_dim = 16, output_dim = 2, heads = 2)
  m
}
#> An `nn_module` containing 5,554 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • input: <nn_linear> #560 parameters
#> • degree_emb: <nn_embedding> #176 parameters
#> • layers: <nn_module_list> #4,480 parameters
#> • ln: <nn_layer_norm> #32 parameters
#> • head: <nn_sequential> #306 parameters
```
