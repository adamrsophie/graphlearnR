# Temporal graph network with node memory

A memory based model for continuous time dynamic link prediction in the
style of TGN (Rossi et al. 2020). Every node keeps a memory vector that
a GRU updates whenever the node takes part in an event. Events are
processed in chronological batches:

## Usage

``` r
tgn_model(
  num_nodes,
  memory_dim = 32L,
  time_dim = 16L,
  embedding_dim = 32L,
  edge_dim = 0L,
  embedding = c("time", "graph_sum"),
  num_neighbors = 10L,
  node_embedding = TRUE
)

train_tgn(
  model,
  split,
  epochs = 10L,
  batch_size = 200L,
  lr = 0.001,
  seed = NULL,
  device = NULL
)
```

## Arguments

- num_nodes:

  Number of nodes.

- memory_dim:

  Memory size.

- time_dim:

  Size of the time encoding.

- embedding_dim:

  Embedding size.

- edge_dim:

  Number of event features (`0` for none).

- embedding:

  `"time"` or `"graph_sum"`.

- num_neighbors:

  Neighbors used by `"graph_sum"`.

- node_embedding:

  Add a learned static embedding per node to \\z_i\\ (useful without
  node features; makes the model transductive).

- model:

  A `tgn_model`.

- split:

  Result of
  [`split_temporal()`](https://adamrsophie.github.io/graphlearnR/reference/split_temporal.md)
  (`train`, `val`, `test`).

- epochs:

  Passes over the training events.

- batch_size:

  Events per batch.

- lr:

  Learning rate.

- seed:

  Optional seed for weights, negatives and the epoch order.

- device:

  Device.

## Value

`tgn_model()`: an `nn_module`.

`train_tgn()`: a `temporal_fit` with `history`, `val_metrics` and
`test_metrics` (ROC AUC and average precision against one random
destination per event). Evaluation continues the memory built over the
training (and, for test, validation) events, without gradient updates.

## Details

1.  The raw messages of the **previous** batch update the memory of the
    nodes involved (message \\\[s_i, s_j, \phi(t - t_i^{last}),
    e\_{ij}\]\\, keeping the most recent message per node, then \\s_i
    \leftarrow \mathrm{GRU}(m_i, s_i)\\). Doing this at the start of the
    next batch lets gradients reach the GRU while the current batch's
    own events never inform its predictions.

2.  Node embeddings are computed from memory and the time since the
    node's last update, \\z_i = W \[s_i, \phi(t - t_i^{last})\]\\
    (`embedding = "time"`), plus, with `embedding = "graph_sum"`, the
    mean of \\W' \[s_j, \phi(t - t\_{ij})\]\\ over the `num_neighbors`
    most recent earlier interaction partners \\j\\.

3.  An MLP scores \\\[z\_{src}, z\_{dst}\]\\; the loss is binary cross
    entropy against one random destination per event.

\\\phi(\Delta t) = \cos(\omega \Delta t + b)\\ is the learnable time
encoding of TGAT (Xu et al. 2020). Memory is detached between batches
(truncated backpropagation over one batch). The graph attention
embedding and the other message aggregators of the paper are not
implemented.

## References

Rossi, Chamberlain, Frasca, Eynard, Monti and Bronstein (2020) Temporal
graph networks for deep learning on dynamic graphs. Xu, Ruan, Korpeoglu,
Kumar and Achan (2020) Inductive representation learning on temporal
graphs.

## Examples

``` r
if (torch::torch_is_installed()) {
  tg <- dataset_synthetic_temporal(seed = 1)
  sp <- split_temporal(tg)
  m <- tgn_model(tg$num_nodes, memory_dim = 16, embedding_dim = 16)
  fit <- train_tgn(m, sp, epochs = 2, batch_size = 100, seed = 1)
  fit$test_metrics
}
#>           roc_auc average_precision 
#>         0.4906086         0.5119871 
```
