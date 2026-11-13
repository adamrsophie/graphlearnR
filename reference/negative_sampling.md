# Negative edge sampling

Draws node pairs that are not edges. Pairs are drawn uniformly from all
allowed pairs by rejection: candidate pairs are sampled uniformly, and
pairs that appear in `exclude`, self loops (unless allowed) and repeats
are discarded. For undirected graphs `(u, v)` and `(v, u)` are the same
pair, each returned pair has `u < v`, and a pair is excluded if either
orientation is in `exclude`.

## Usage

``` r
negative_sampling(
  exclude,
  num_nodes = NULL,
  num_samples,
  directed = TRUE,
  allow_self_loops = FALSE,
  num_src = NULL,
  num_dst = NULL,
  seed = NULL,
  max_tries = 100L
)
```

## Arguments

- exclude:

  2 by E matrix of known positive pairs to avoid (all known positives,
  including validation and test edges, for the usual filtered protocol).

- num_nodes:

  Number of nodes (homogeneous graphs).

- num_samples:

  Number of negative pairs.

- directed:

  Treat pairs as ordered.

- allow_self_loops:

  Allow `(u, u)` pairs.

- num_src, num_dst:

  Node counts for bipartite sampling.

- seed:

  Optional integer seed (see
  [`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md)).

- max_tries:

  Rejection rounds before giving up.

## Value

A 2 by `num_samples` integer matrix.

## Details

For bipartite or typed relations give `num_src` and `num_dst`; pairs are
then (source of type A, target of type B) and self loops are not
special.

## Examples

``` r
negative_sampling(rbind(c(1, 2), c(2, 3)), num_nodes = 4, num_samples = 3,
                  seed = 1)
#>      [,1] [,2] [,3]
#> [1,]    4    3    3
#> [2,]    2    2    1
```
