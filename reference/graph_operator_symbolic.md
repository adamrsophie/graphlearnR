# Exact graph operators with symcompR

Returns a graph operator as an exact symbolic matrix from
[symcompR](https://github.com/adamrsophie/symcompR): the adjacency
matrix, the Laplacian \\D - A\\, or the GCN propagation matrix \\\hat
D^{-1/2} (A + I) \hat D^{-1/2}\\ used by
[`gcn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md),
with square roots kept exact. Row \\i\\, column \\j\\ is the weight from
\\j\\ to \\i\\. Edge weights are ignored (unweighted operators), so
every entry is an integer, a rational or an algebraic number.

## Usage

``` r
graph_operator_symbolic(g, operator = c("adjacency", "laplacian", "gcn"))
```

## Arguments

- g:

  A `graph_data` object.

- operator:

  `"adjacency"`, `"laplacian"` or `"gcn"`.

## Value

A symcompR matrix; [`as.matrix()`](https://rdrr.io/r/base/matrix.html)
gives its numeric value.

## Details

This is an optional bridge, useful for checking layer normalizations
against exact values or showing the operator a layer applies. It needs
symcompR and is meant for small graphs.

## Examples

``` r
if (requireNamespace("symcompR", quietly = TRUE)) {
  g <- graph_data(edge_index = rbind(1:2, 2:3), num_nodes = 3, directed = FALSE)
  graph_operator_symbolic(g, "gcn")
}
#> Matrix(3, 3)
#> [1/2      , sqrt(6)/6, 0        ]
#> [sqrt(6)/6, 1/3      , sqrt(6)/6]
#> [0        , sqrt(6)/6, 1/2      ]
```
