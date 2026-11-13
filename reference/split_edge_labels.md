# Split labeled edges for edge classification or regression

Sets logical masks `edge_train_mask`, `edge_val_mask` and
`edge_test_mask` over stored edges. Both directions of an undirected
pair get the same assignment. Only edges with a non missing `edge_y` are
assigned. The graph structure stays intact: edge labels, not edges, are
held out.

## Usage

``` r
split_edge_labels(g, train = 0.6, val = 0.2, test = 0.2, seed = NULL)
```

## Arguments

- g:

  A `graph_data` object with node labels `y`.

- train, val, test:

  Proportions (summing to at most 1) or counts.

- seed:

  Optional integer seed; the caller's random state is not changed.

## Value

`g` with the edge masks set.
