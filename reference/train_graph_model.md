# Train a graph neural network

A training loop for the common graph learning tasks. It uses the model,
optimizer and loss as ordinary `torch` objects, so all of them stay
accessible in the returned fit.

## Usage

``` r
train_graph_model(
  model,
  data,
  task = c("node_classification", "node_regression", "graph_classification",
    "graph_regression", "link_prediction", "edge_classification", "edge_regression"),
  epochs = 100L,
  optimizer = "adam",
  lr = 0.01,
  weight_decay = 0,
  loss = NULL,
  val_data = NULL,
  batch_size = NULL,
  fanouts = NULL,
  decoder = NULL,
  scheduler = NULL,
  early_stopping = NULL,
  checkpoint_path = NULL,
  device = NULL,
  seed = NULL,
  grad_clip = NULL,
  class_weights = NULL,
  target_type = NULL,
  loader = NULL,
  verbose = FALSE
)
```

## Arguments

- model:

  An `nn_module` whose forward takes `(x, edge_index, ...)` with the
  named arguments documented in
  [`gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/gnn_model.md).

- data:

  Training data; see the table.

- task:

  Prediction task.

- epochs:

  Number of epochs.

- optimizer:

  `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`, `"adagrad"`, or a function
  taking a parameter list and returning a `torch` optimizer.

- lr:

  Learning rate.

- weight_decay:

  L2 penalty passed to the optimizer.

- loss:

  `NULL` for the task default (cross entropy, mean squared error, binary
  cross entropy with logits), a name (`"cross_entropy"`, `"mse"`,
  `"l1"`, `"huber"`, `"bce"`, `"focal"`), or a function
  `(prediction, target)`.

- val_data:

  Validation graphs for graph level tasks.

- batch_size:

  Graphs per batch (graph tasks) or seeds per batch (node tasks with
  `fanouts`).

- fanouts:

  Neighbor sampling fanouts for node mini-batches.

- decoder:

  Decoder module for link and edge tasks; defaults to
  [`link_decoder()`](https://adamrsophie.github.io/graphlearnR/reference/link_decoder.md)
  `"dot"` and an
  [`edge_decoder()`](https://adamrsophie.github.io/graphlearnR/reference/link_decoder.md).

- scheduler:

  `NULL`, a list such as
  `list(type = "step", step_size = 50, gamma = 0.5)`,
  `list(type = "plateau", patience = 10, factor = 0.5)` or
  `list(type = "cosine")`, or a function taking the optimizer and
  returning a `torch` scheduler.

- early_stopping:

  Patience in epochs, or `NULL` to train all epochs. Monitors validation
  loss; the best weights are restored at the end.

- checkpoint_path:

  Optional file; the model state is saved there with
  [`torch::torch_save()`](https://torch.mlverse.org/docs/reference/torch_save.html)
  whenever validation loss improves.

- device:

  Device, see
  [`graph_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/graph_tensors.md).

- seed:

  Optional integer seed.

- grad_clip:

  Optional maximum gradient norm.

- class_weights:

  Optional class weights for cross entropy.

- target_type:

  Node type to predict for heterogeneous graphs.

- loader:

  A
  [`cluster_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
  or
  [`saint_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
  for node tasks.

- verbose:

  Print progress every `verbose` epochs (`FALSE` for none).

## Value

An object of class `graph_fit`.

## Details

|  |  |  |  |
|----|----|----|----|
| `task` | `data` | targets | loss on |
| `"node_classification"`, `"node_regression"` | `graph_data` (or `hetero_graph_data` with `target_type`) with `train_mask` | node `y` | training nodes only |
| `"graph_classification"`, `"graph_regression"` | list of `graph_data` or a `graph_dataset`; `val_data` likewise | `graph_y` | every training graph |
| `"link_prediction"` | a `link_split` from [`split_edges()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges.md) (or [`split_edges_hetero()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges_hetero.md)) | positive and sampled negative pairs | training pairs |
| `"edge_classification"`, `"edge_regression"` | `graph_data` with `edge_y` and masks from [`split_edge_labels()`](https://adamrsophie.github.io/graphlearnR/reference/split_edge_labels.md) | edge `y` | training edges |

**Node mini-batches.** With `fanouts`, node tasks are trained on sampled
neighborhoods
([`node_loader()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md));
the loss uses only the seed nodes of each batch, never the sampled
context nodes. Validation uses the full graph.

**Subgraph batches.** With `loader` from
[`cluster_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
or
[`saint_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md),
node tasks are trained on induced subgraphs; the loss uses the training
nodes in each batch, weighted by GraphSAINT's `node_norm` when present.

**Link prediction.** Message passing uses `split$train_graph`, which
holds training edges only. Fresh negative pairs are drawn every epoch,
excluding training positives; validation and test pairs are never used
for training.

**Reproducibility.** With `seed`, `torch`'s global generator is seeded
with `torch_manual_seed(seed)` (needed for weight initialization done
before the call, and for dropout), and every R side random draw
(batching, neighbor and negative sampling) uses a seed derived from
`seed` inside
[`withr::with_seed()`](https://withr.r-lib.org/reference/with_seed.html),
so the caller's R random state is not changed. The fit records seeds,
settings, package versions and the device in `fit$reproducibility`.
Training on GPUs may still be nondeterministic, because scatter
additions on CUDA use atomic operations whose order is not fixed.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  model <- gnn_model("sage", input_dim = num_node_features(g),
                     hidden_dim = 16, output_dim = 2)
  fit <- train_graph_model(model, g, "node_classification", epochs = 30,
                           seed = 1)
  fit
  head(predict(fit, g))
}
#>   .row node_id .pred_class .prob_Mr. Hi .prob_Officer
#> 1    1       1      Mr. Hi    0.9628773    0.03712269
#> 2    2       2      Mr. Hi    0.9669216    0.03307841
#> 3    3       3      Mr. Hi    0.9286499    0.07135003
#> 4    4       4      Mr. Hi    0.9844617    0.01553830
#> 5    5       5      Mr. Hi    0.9754194    0.02458063
#> 6    6       6      Mr. Hi    0.9762313    0.02376873
```
