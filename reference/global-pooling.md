# Global pooling (graph readout)

Reduce node features to one row per graph using the membership vector
`batch` of a
[`graph_batch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
(values `1:num_graphs`). Sum, mean and max are permutation invariant:
reordering nodes within a graph does not change the result.

## Usage

``` r
global_add_pool(x, batch = NULL, num_graphs = NULL)

global_mean_pool(x, batch = NULL, num_graphs = NULL)

global_max_pool(x, batch = NULL, num_graphs = NULL)
```

## Arguments

- x:

  Node feature tensor (N by F).

- batch:

  Long tensor or integer vector of graph membership. `NULL` treats all
  nodes as one graph.

- num_graphs:

  Number of graphs; inferred from `batch` when `NULL`.

## Value

A tensor with one row per graph.

## Examples

``` r
if (torch::torch_is_installed()) {
  x <- torch::torch_tensor(matrix(1:6, 3, 2), dtype = torch::torch_float())
  b <- c(1L, 1L, 2L)
  global_add_pool(x, b)
  global_mean_pool(x, b)
  global_max_pool(x, b)
}
#> torch_tensor
#>  2  5
#>  3  6
#> [ CPUFloatType{2,2} ]
```
