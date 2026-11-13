# Corrupted tails for ranking evaluation

For each positive pair `(u, v)`, draws `k` candidate targets `v'` such
that `(u, v')` is not a known positive (filtered setting, Bordes et al.
2013) unless `filtered = FALSE`. Used for mean reciprocal rank and
per-query Hits@K.

## Usage

``` r
corrupt_tails(
  pos,
  num_nodes,
  k,
  known = pos,
  filtered = TRUE,
  directed = TRUE,
  seed = NULL
)
```

## Arguments

- pos:

  2 by K matrix of positive pairs.

- num_nodes:

  Number of candidate targets.

- k:

  Negatives per positive.

- known:

  2 by E matrix of all known positives (filtered setting).

- filtered:

  Exclude known positives.

- directed:

  With `FALSE`, a known pair excludes both orientations.

- seed:

  Optional integer seed.

## Value

A K by `k` integer matrix of negative targets. The head of a pair is
never drawn as its own corrupted tail.

## Examples

``` r
pos <- rbind(c(1, 2), c(2, 3))
corrupt_tails(pos, num_nodes = 10, k = 3, seed = 1)
#>      [,1] [,2] [,3]
#> [1,]    3    6   10
#> [2,]    4    7    5
```
