# Dense batch layout

Reorganizes node rows of a batch into a padded `B` by `L` by `F` tensor,
where `L` is the largest graph size, plus a `B` by `L` logical mask of
real nodes. `from_dense_batch()` inverts it.

## Usage

``` r
to_dense_batch(x, batch)

from_dense_batch(dense)
```

## Arguments

- x:

  Node feature tensor (N by F).

- batch:

  Graph membership (integer vector, `1:B`, nodes of a graph contiguous
  as produced by
  [`graph_batch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)).

- dense:

  Result of `to_dense_batch()` with `x` replaced by a new `B` by `L` by
  `F'` tensor.

## Value

A list with `x`, `mask` and `index` (the flat positions).

## Examples

``` r
if (torch::torch_is_installed()) {
  x <- torch::torch_randn(5, 2)
  d <- to_dense_batch(x, batch = c(1, 1, 2, 2, 2))
  d$x$shape
  from_dense_batch(d)$shape
}
#> [1] 5 2
```
