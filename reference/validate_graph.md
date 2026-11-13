# Validate a graph data object

Checks that every stored field is consistent with the number of nodes
and the number of stored edges: edge indices are whole numbers in
`1:num_nodes`, feature matrices have one row per node or edge, masks are
logical vectors of the right length, and labels have one entry per node
or edge. Problems are reported together in one error.

## Usage

``` r
validate_graph(g)
```

## Arguments

- g:

  A `graph_data` object.

## Value

`g`, invisibly, if it is valid.

## Details

Self loops, duplicate edges (multigraphs) and isolated nodes are valid;
query them with
[`has_self_loops()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md),
[`has_duplicate_edges()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
and
[`has_isolated_nodes()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md).

## Examples

``` r
g <- graph_data(edge_index = rbind(1:2, 2:3))
validate_graph(g)
try(graph_data(edge_index = rbind(1, 5), num_nodes = 3))
#> Error : Invalid graph_data:
#> * `edge_index` refers to nodes outside 1:3 (first bad edge: column 1, 1 -> 5).
```
