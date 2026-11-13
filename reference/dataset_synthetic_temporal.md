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
