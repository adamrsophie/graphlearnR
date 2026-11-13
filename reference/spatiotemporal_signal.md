# Spatiotemporal signals on a fixed graph

A graph whose structure is fixed while node features change over time,
such as traffic sensors on a road network.

## Usage

``` r
spatiotemporal_signal(graph, features, targets = NULL, times = NULL)
```

## Arguments

- graph:

  A `graph_data` object (its `x` is ignored).

- features:

  Array `T` by `N` by `F` (or a `T` by `N` matrix for one feature).

- targets:

  Optional `T` by `N` matrix of targets; default the first feature.

- times:

  Optional time stamps.

## Value

An object of class `spatiotemporal_signal`.

## Examples

``` r
g <- dataset_karate()
sig <- spatiotemporal_signal(g, matrix(rnorm(10 * 34), 10, 34))
sig
#> <spatiotemporal_signal> 10 steps, 34 nodes, 1 features
```
