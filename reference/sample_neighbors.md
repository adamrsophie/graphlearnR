# Multi-hop neighbor sampling

Samples the computation graph of a set of seed nodes, as in GraphSAGE
(Hamilton et al. 2017) and PyG's `NeighborLoader`. Hop `k` samples up to
`fanouts[k]` **incoming** edges of every node reached in hop `k - 1`
(hop 0 is the seed set); sources of the sampled edges join the subgraph.
Messages in a GNN flow along these edges towards the seeds, so an `L`
layer model with `L` fanouts sees exactly the information it would see
on the full graph when every fanout is `-1` (all neighbors).

## Usage

``` r
sample_neighbors(g, seeds, fanouts, replace = FALSE, seed = NULL)
```

## Arguments

- g:

  A `graph_data` object.

- seeds:

  Integer vector of seed node indices (one based).

- fanouts:

  Integer vector, one entry per hop; `-1` keeps all incoming edges.

- replace:

  Sample edges with replacement.

- seed:

  Optional integer seed.

## Value

A `graph_data` object of class `sampled_graph`.

## Details

The result is a `graph_data` object with local indices and these extra
fields:

- `n_id`: original index of every local node; the seeds come first, in
  the order given, so `seed_index = seq_along(seeds)`;

- `e_id`: original stored edge index of every sampled edge;

- `hop`: hop at which each node was first reached (0 for seeds);

- `num_seeds`.

Losses and metrics must use only the seed rows (`seed_index`); the other
nodes are context.
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
does this automatically. The sampled graph is marked directed because
sampling may keep only one direction of an undirected pair.

Sampling uses R's random number generator. With `seed`, it runs inside
[`withr::with_seed()`](https://withr.r-lib.org/reference/with_seed.html),
so the result is reproducible and the caller's random state is left
untouched.

## Examples

``` r
g <- dataset_karate()
sg <- sample_neighbors(g, seeds = c(1, 34), fanouts = c(3, 2), seed = 1)
sg$num_seeds
#> [1] 2
sg$n_id
#>  [1]  1 34  6 12 14 19 21 27 11  2  4 33 30
```
