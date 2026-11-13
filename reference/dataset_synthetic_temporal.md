# Synthetic temporal data

`dataset_synthetic_temporal()` simulates an interaction stream on a
community graph where pairs that interacted before tend to interact
again, so the past predicts the future. `dataset_synthetic_signal()`
simulates a diffusion process on a ring graph: each node's value moves
towards the mean of its neighbors plus noise.

## Usage

``` r
dataset_synthetic_temporal(num_nodes = 40L, num_events = 800L, seed = NULL)

dataset_synthetic_signal(num_nodes = 20L, num_steps = 60L, seed = NULL)
```

## Arguments

- num_nodes:

  Nodes.

- num_events:

  Events.

- seed:

  Optional integer seed.

- num_steps:

  Time steps.

## Value

A `temporal_graph_data` or a `spatiotemporal_signal`.

## Examples

``` r
tg <- dataset_synthetic_temporal(seed = 1)
tg
#> <temporal_graph_data> 40 nodes, 800 events, t in [0.1478096, 99.91947]
sig <- dataset_synthetic_signal(seed = 1)
dim(sig$features)
#> [1] 60 20  1
```
