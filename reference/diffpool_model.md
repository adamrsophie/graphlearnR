# DiffPool graph classification model

Two
[`graph_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
stacks (sum aggregation) compute node embeddings and soft cluster
assignments on the sparse batch;
[`dense_diff_pool()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_adj.md)
coarsens each graph to `num_clusters` nodes; a
[`dense_graph_conv()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_adj.md)
runs on the coarsened graphs; the cluster embeddings are averaged and
passed to a linear head. After each forward pass the sum of the link and
entropy losses is stored in `model$aux_loss`, which
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
adds to the task loss.

## Usage

``` r
diffpool_model(
  input_dim,
  hidden_dim = 32L,
  output_dim = 2L,
  num_clusters = 8L,
  num_layers = 2L,
  aux_weight = 1
)
```

## Arguments

- input_dim:

  Node feature size.

- hidden_dim:

  Hidden size.

- output_dim:

  Output size (classes or targets).

- num_clusters:

  Clusters per graph after pooling.

- num_layers:

  Message passing layers before pooling.

- aux_weight:

  Weight of the auxiliary losses.

## Value

An `nn_module`.

## Examples

``` r
if (torch::torch_is_installed()) {
  gs <- dataset_synthetic_graphs(40, seed = 1)
  m <- diffpool_model(1, 16, 2, num_clusters = 4)
  fit <- train_graph_model(m, gs, "graph_classification", epochs = 3, seed = 1)
}
```
