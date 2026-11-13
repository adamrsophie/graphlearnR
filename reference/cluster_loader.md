# Cluster and GraphSAINT subgraph loaders

Loaders that train node models on whole subgraphs instead of sampled
neighborhoods. Pass one to
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
as `loader`; every batch is an induced subgraph, the loss uses the
training nodes inside it, and validation uses the full graph.

## Usage

``` r
cluster_loader(
  g,
  num_parts = 10L,
  clusters_per_batch = 1L,
  method = c("louvain", "random"),
  shuffle = TRUE,
  seed = NULL
)

saint_loader(
  g,
  batch_size = 100L,
  walk_length = 2L,
  num_steps = 10L,
  sample_coverage = 20L,
  seed = NULL
)
```

## Arguments

- g:

  A `graph_data` object.

- num_parts:

  Number of parts.

- clusters_per_batch:

  Parts combined into one batch.

- method:

  `"louvain"` or `"random"`.

- shuffle:

  Shuffle parts every epoch.

- seed:

  Optional seed; the caller's random stream is left untouched.

- batch_size:

  Number of random walk roots per batch.

- walk_length:

  Steps per walk.

- num_steps:

  Batches per epoch.

- sample_coverage:

  Number of subgraphs pre-sampled per node (on average) to estimate
  normalization; `0` disables normalization.

## Value

A loader of class `subgraph_loader`;
[`loader_batches()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md)
returns a list of `graph_data` batches whose `node_ids` hold original
node indices.

## Details

- `cluster_loader()` follows Cluster-GCN (Chiang et al. 2019): the graph
  is partitioned once into `num_parts` parts and each batch is the
  subgraph induced by `clusters_per_batch` randomly chosen parts,
  including the edges between them. With `method = "louvain"`, nodes are
  ordered by Louvain community (igraph's `cluster_louvain()` on the
  undirected view) and then by breadth first search order, and the
  ordering is cut into `num_parts` parts of equal size, so parts follow
  communities where sizes allow; `method = "random"` assigns nodes at
  random. METIS is not used.

- `saint_loader()` follows the GraphSAINT random walk sampler (Zeng et
  al. 2020): each batch is the subgraph induced by the nodes visited by
  `batch_size` random walks of length `walk_length` from uniformly drawn
  roots. Normalization coefficients are estimated from `sample_coverage`
  pre-sampled subgraphs, as in PyTorch Geometric's `GraphSAINTSampler`:
  each batch carries `node_norm` (loss weight \\\propto 1 / p_v\\) and
  edge weights multiplied by \\p_v / p\_{uv}\\ (aggregator
  normalization), where \\p\\ are empirical inclusion frequencies. The
  trainer sums the per node loss of the training nodes in a batch,
  weighted by `node_norm`, for the built in losses and for loss
  functions with a `reduction` argument; in expectation this is the mean
  loss over all nodes of the graph.

Batches are drawn in R on the CPU.

## References

Chiang et al. (2019) Cluster-GCN. Zeng et al. (2020) GraphSAINT: graph
sampling based inductive learning method.

## Examples

``` r
g <- split_nodes(dataset_sbm(c(40, 40, 40), seed = 1), seed = 1)
cl <- cluster_loader(g, num_parts = 6, clusters_per_batch = 2, seed = 1)
lengths(cl$parts)
#> [1] 20 20 20 20 20 20
vapply(loader_batches(cl), num_nodes, 1L)
#> [1] 40 40 40
sl <- saint_loader(g, batch_size = 10, walk_length = 3, seed = 1)
summary(loader_batches(sl)[[1]]$node_norm)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.01625 0.02708 0.03095 0.03255 0.03824 0.06500 
```
