# Build a graph neural network

Stacks `num_layers` message passing layers of one architecture, each
followed by an activation, optional normalization and dropout, then adds
a prediction head:

## Usage

``` r
gnn_model(
  architecture = c("sage", "gcn", "gat", "gatv2", "gin", "graph_conv", "transformer",
    "mpnn", "rgcn"),
  input_dim,
  hidden_dim = 64L,
  output_dim = NULL,
  num_layers = 2L,
  dropout = 0.5,
  activation = "relu",
  heads = 4L,
  edge_dim = NULL,
  aggr = NULL,
  pool = NULL,
  norm = c("none", "batch", "layer"),
  residual = FALSE,
  num_relations = NULL,
  num_bases = NULL
)
```

## Arguments

- architecture:

  One of `"sage"`, `"gcn"`, `"gat"`, `"gatv2"`, `"gin"`, `"graph_conv"`,
  `"transformer"`, `"mpnn"` (edge conditioned
  [`nn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)),
  `"rgcn"`.

- input_dim:

  Node feature size.

- hidden_dim:

  Hidden size. For `"gat"`, `"gatv2"` and `"transformer"` it must be
  divisible by `heads`.

- output_dim:

  Output size (classes or targets), or `NULL`.

- num_layers:

  Number of message passing layers.

- dropout:

  Dropout probability after each layer.

- activation:

  Activation name: `"relu"`, `"elu"`, `"gelu"`, `"tanh"` or
  `"leaky_relu"` (slope 0.2).

- heads:

  Attention heads.

- edge_dim:

  Edge feature size (`"mpnn"`, GINE, attention with edge features).

- aggr:

  Neighbor aggregation for `"sage"`, `"graph_conv"`, `"mpnn"`, `"rgcn"`.

- pool:

  `NULL`, `"mean"`, `"sum"`, `"max"` or `"attention"`.

- norm:

  `"none"`, `"batch"` or `"layer"` normalization after each layer.

- residual:

  Add a residual connection around each layer after the first.

- num_relations, num_bases:

  Relation count and bases for `"rgcn"`.

## Value

An `nn_module` of class `graph_model`.

## Details

- node level (`pool = NULL`): a linear layer to `output_dim` per node;

- graph level (`pool` set): global pooling, then an MLP to `output_dim`
  per graph;

- encoder only (`output_dim = NULL`): returns node embeddings of size
  `hidden_dim`, for link prediction or representation learning.

The model's
`forward(x, edge_index, edge_weight, edge_attr, batch, num_graphs, edge_type)`
takes tensors from
[`graph_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/graph_tensors.md).
Setting `return_embedding = TRUE` returns the last hidden node
representation instead of the head output. The construction arguments
are kept in `model$config` so that
[`save_graph_fit()`](https://adamrsophie.github.io/graphlearnR/reference/save_graph_fit.md)
can rebuild the model.

## Examples

``` r
if (torch::torch_is_installed()) {
  m <- gnn_model("sage", input_dim = 4, hidden_dim = 8, output_dim = 3)
  x <- torch::torch_randn(5, 4)
  ei <- torch::torch_tensor(rbind(1:4, 2:5), dtype = torch::torch_long())
  m(x, ei)$shape
}
#> [1] 5 3
```
