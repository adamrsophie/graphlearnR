# Basic graph queries

Basic graph queries

## Usage

``` r
num_nodes(g)

num_edges(g, unique_pairs = FALSE)

num_node_features(g)

num_edge_features(g)

is_directed_graph(g)

has_self_loops(g)

has_isolated_nodes(g)

has_duplicate_edges(g)

degree(g, mode = c("in", "out"), weighted = FALSE)
```

## Arguments

- g:

  A `graph_data`, `graph_batch` or `hetero_graph_data` object.

- unique_pairs:

  For undirected graphs, count each undirected edge once instead of
  counting stored directed entries.

- mode:

  `"in"` counts incoming edges (edges whose target is the node), `"out"`
  counts outgoing edges. For undirected graphs both agree.

- weighted:

  Sum edge weights instead of counting edges.

## Value

An integer (or logical) scalar; `degree()` returns an integer vector.

## Examples

``` r
g <- graph_data(edge_index = rbind(c(1, 2), c(2, 3)), directed = FALSE)
num_nodes(g)
#> [1] 3
num_edges(g)
#> [1] 4
num_edges(g, unique_pairs = TRUE)
#> [1] 2
degree(g)
#> [1] 1 2 1
```
