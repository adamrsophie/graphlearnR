# Convert between igraph and graph_data

`from_igraph()` builds a
[`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md)
object from an `igraph` graph. Vertex order is kept, so vertex `i`
becomes node `i`. Vertex names (or a `node_id` vertex attribute written
by `to_igraph()`) become `node_ids`. Numeric vertex attributes named in
`x_attrs` become node feature columns in that order; edge attributes
named in `edge_attrs` become edge feature columns.

## Usage

``` r
from_igraph(
  ig,
  x_attrs = NULL,
  y_attr = "y",
  edge_attrs = NULL,
  weight_attr = "weight",
  edge_y_attr = "edge_y"
)

to_igraph(g)
```

## Arguments

- ig:

  An `igraph` object.

- x_attrs:

  Character vector of numeric vertex attributes to use as node features.
  `NULL` picks attributes named `x_1`, `x_2`, ... if any.

- y_attr:

  Vertex attribute holding node labels (`NULL` for none). The default
  uses `y` when present.

- edge_attrs:

  Edge attributes to use as edge features. `NULL` picks `e_1`, `e_2`,
  ... if any.

- weight_attr:

  Edge attribute holding weights, used when present.

- edge_y_attr:

  Edge attribute holding edge labels, used when present.

- g:

  A `graph_data` object.

## Value

`from_igraph()` returns a `graph_data` object; `to_igraph()` returns an
`igraph` object.

## Details

`to_igraph()` writes features as attributes `x_1`, `x_2`, ... on
vertices and `e_1`, `e_2`, ... on edges, labels as `y` and `edge_y`,
weights as `weight`, and the original identifiers as `node_id`. With the
default arguments, `from_igraph()` reads these back, so the round trip
keeps node order, identifiers, features and their column names, node,
edge and graph labels (multi target label matrices as `y_1`, `y_2`,
...), weights, types, node and edge masks, `graph_attr`, `metadata` and
direction. Graph level fields are stored as igraph graph attributes.

For undirected graphs `to_igraph()` writes each undirected pair once.

## Examples

``` r
ig <- igraph::make_ring(5)
g <- from_igraph(ig)
g
#> <graph_data> undirected graph: 5 nodes, 10 edges (5 undirected pairs)
ig2 <- to_igraph(g)
igraph::ecount(ig2)
#> [1] 5
```
