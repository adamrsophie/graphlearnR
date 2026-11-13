# Batch heterogeneous graphs

Concatenates graphs with the same node and relation types. Within each
node type, indices of graph `k` are shifted by the node counts of that
type in graphs `1, ..., k - 1`; `batch[[type]]` records graph
membership.

## Usage

``` r
hetero_batch(graphs)
```

## Arguments

- graphs:

  A list of `hetero_graph_data` objects.

## Value

A `hetero_graph_data` of class `hetero_graph_batch`.
