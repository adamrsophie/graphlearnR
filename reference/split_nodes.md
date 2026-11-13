# Split nodes into training, validation and test sets

Sets `train_mask`, `val_mask` and `test_mask` on a graph for
transductive node prediction. Proportions or counts are accepted. With
`stratify`, each class is split separately so class proportions are
kept. Nodes with a missing label are never assigned.

## Usage

``` r
split_nodes(
  g,
  train = 0.6,
  val = 0.2,
  test = 0.2,
  stratify = is.factor(g$y),
  seed = NULL
)
```

## Arguments

- g:

  A `graph_data` object with node labels `y`.

- train, val, test:

  Proportions (summing to at most 1) or counts.

- stratify:

  Split each class separately (factor labels only).

- seed:

  Optional integer seed; the caller's random state is not changed.

## Value

`g` with the three masks set.

## Details

Transductive splits share one graph: the features and edges of test
nodes are visible during training, only their labels are held out. This
is the standard protocol for citation benchmarks (Kipf and Welling
2017); for inductive evaluation split graphs with
[`split_graphs()`](https://adamrsophie.github.io/graphlearnR/reference/split_graphs.md)
instead.

## Examples

``` r
g <- split_nodes(dataset_karate(), train = 0.5, val = 0.25, test = 0.25,
                 seed = 1)
sum(g$train_mask)
#> [1] 18
```
