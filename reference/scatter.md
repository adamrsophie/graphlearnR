# Scatter aggregation over groups

Reduce rows of `src` into `dim_size` groups given by `index`: row `k` of
the result combines all rows `j` with `index[j] == k`. These are the
aggregation kernels used by
[`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md);
they are vectorized `torch` operations (`index_add`, `scatter_reduce`)
and differentiable.

## Usage

``` r
scatter_sum(src, index, dim_size)

scatter_count(index, dim_size, dtype = torch_float(), device = NULL)

scatter_mean(src, index, dim_size)

scatter_max(src, index, dim_size)

scatter_min(src, index, dim_size)

scatter_std(src, index, dim_size, eps = 1e-05)

scatter_softmax(src, index, dim_size)

scatter_aggregate(src, index, dim_size, aggr = "sum")
```

## Arguments

- src:

  Tensor with one row per element (for example per edge).

- index:

  One based long tensor (or integer vector) of group ids, one per row of
  `src`.

- dim_size:

  Number of groups (for example the number of nodes).

- dtype, device:

  Dtype and device of the count tensor.

- eps:

  Variance floor in `scatter_std()`; smaller variances give 0.

- aggr:

  One or more of `"sum"` (alias `"add"`), `"mean"`, `"max"`, `"min"`,
  `"std"`.

## Value

A tensor with `dim_size` rows.

## Details

- `scatter_sum()`: sum of rows. Empty groups give 0.

- `scatter_mean()`: sum divided by the group size. Empty groups give 0.

- `scatter_max()`, `scatter_min()`: elementwise maximum or minimum.
  Empty groups give 0. Gradients flow to the selected entries; ties
  split the gradient as `torch` defines for `scatter_reduce`.

- `scatter_std()`: population standard deviation, computed as
  \\\sqrt{\max(\mathrm{var}, \epsilon)}\\ and set to 0 where the
  variance is at most \\\epsilon\\ (empty or constant groups), as in
  PyTorch Geometric's `StdAggregation`; the gradient stays finite.

- `scatter_softmax()`: softmax of `src` within each group (used for
  attention over incoming edges); columns are treated separately.

- `scatter_aggregate()`: dispatch by name; several names give their
  results concatenated column wise.

## Examples

``` r
if (torch::torch_is_installed()) {
  src <- torch::torch_tensor(matrix(c(1, 2, 3, 4), 4, 1))
  idx <- torch::torch_tensor(c(1L, 1L, 2L, 2L), dtype = torch::torch_long())
  scatter_sum(src, idx, 3)
  scatter_mean(src, idx, 3)
  scatter_max(src, idx, 3)
}
#> torch_tensor
#>  2
#>  4
#>  0
#> [ CPUFloatType{3,1} ]
```
