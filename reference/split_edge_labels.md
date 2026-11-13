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

  Proportions (each at most 1, summing to at most 1) or whole counts.
  Values that are all at most 1 are read as proportions, so
  `train = 1, val = 0, test = 0` puts every labeled node in training.
  With `stratify`, sizes apply within each class.

- seed:

  Optional integer seed; the caller's random state is not changed.

## Value

`g` with the edge masks set.

## Examples

``` r
g <- dataset_karate()
g$edge_y <- factor(ifelse(g$edge_index[1, ] < g$edge_index[2, ], "up", "down"))
g <- split_edge_labels(g, seed = 1)
table(g$edge_train_mask)
#> 
#> FALSE  TRUE 
#>    62    94 
```
