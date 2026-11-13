# Build a graph from an adjacency matrix

Every nonzero entry `A[i, j]` becomes an edge from node `i` to node `j`
with weight `A[i, j]`. For an undirected graph only the upper triangle
(including the diagonal) is read, so each pair is created once and
stored in both directions. Dense matrices and `Matrix` sparse matrices
are accepted.

## Usage

``` r
graph_from_adjacency(A, directed = NULL, weighted = TRUE, ...)

adjacency_from_graph(g, sparse = TRUE, weighted = TRUE)
```

## Arguments

- A:

  Square numeric matrix or `Matrix` object.

- directed:

  `NULL` treats a symmetric matrix as undirected and any other matrix as
  directed.

- weighted:

  Keep entries as edge weights.

- ...:

  Passed to
  [`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md)
  (for example `x` and `y`).

- g:

  A `graph_data` object.

- sparse:

  Return a `Matrix` sparse matrix instead of a dense one.

## Value

A `graph_data` object.

## Details

`adjacency_from_graph()` sums weights of duplicate edges. Without
weights every edge counts 1.

## Examples

``` r
A <- matrix(c(0, 1, 0,
              1, 0, 2,
              0, 2, 0), 3, byrow = TRUE)
g <- graph_from_adjacency(A)
g$edge_weight
#> [1] 1 1 2 2
as.matrix(adjacency_from_graph(g))
#>      [,1] [,2] [,3]
#> [1,]    0    1    0
#> [2,]    1    0    2
#> [3,]    0    2    0
```
