# K-hop subgraph

Exact `k` hop computation graph of `nodes`: all incoming edges up to `k`
hops, equivalent to
[`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md)
with every fanout `-1`.

## Usage

``` r
k_hop_subgraph(g, nodes, k)
```

## Arguments

- g:

  A `graph_data` object.

- nodes:

  Seed nodes.

- k:

  Number of hops.

## Value

A `sampled_graph`.

## Examples

``` r
sub <- k_hop_subgraph(dataset_karate(), nodes = 1, k = 1)
sub$num_nodes
#> [1] 17
```
