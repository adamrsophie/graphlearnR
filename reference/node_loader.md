# Mini-batch loaders

`node_loader()` splits seed nodes into batches and samples each batch's
neighborhood with
[`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md).
`graph_loader()` splits a list of graphs (or a
[`graph_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md))
into
[`graph_batch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
objects. Call `loader_batches()` to draw the batches of one epoch; with
a `seed`, epoch `e` uses seed `seed + e`, so every epoch differs but
runs are reproducible.

## Usage

``` r
node_loader(
  g,
  nodes = NULL,
  batch_size = 64L,
  fanouts = c(10L, 10L),
  shuffle = TRUE,
  seed = NULL
)

graph_loader(dataset, batch_size = 32L, shuffle = TRUE, seed = NULL)

loader_batches(loader, epoch = 1L)
```

## Arguments

- g:

  A `graph_data` object.

- nodes:

  Seed nodes; default the training mask (or all nodes).

- batch_size:

  Seeds or graphs per batch.

- fanouts:

  Fanout per hop, see
  [`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md).

- shuffle:

  Shuffle before splitting into batches.

- seed:

  Optional integer seed.

- dataset:

  A list of `graph_data` objects or a `graph_dataset`.

- loader:

  A loader.

- epoch:

  Epoch number, used to derive the epoch seed.

## Value

A loader object; use `loader_batches()` to get batches.

## Examples

``` r
g <- dataset_karate()
ld <- node_loader(g, nodes = 1:20, batch_size = 8, fanouts = c(5, 5), seed = 1)
length(loader_batches(ld, epoch = 1))
#> [1] 3
```
