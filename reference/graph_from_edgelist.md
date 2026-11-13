# Build a graph from an edge list

Accepts edges that refer to external node identifiers (character or
numeric, not necessarily consecutive) and maps them to one based node
indices. Node order follows `nodes[[id_col]]` when a node table is
given, and otherwise the order of first appearance in `from` then `to`.

## Usage

``` r
graph_from_edgelist(
  edges,
  from = "from",
  to = "to",
  nodes = NULL,
  id_col = "id",
  feature_cols = NULL,
  label_col = NULL,
  edge_feature_cols = NULL,
  weight_col = NULL,
  edge_label_col = NULL,
  directed = TRUE
)

edgelist_from_graph(g)
```

## Arguments

- edges:

  Data frame with one row per edge.

- from, to:

  Column names in `edges` holding source and target identifiers.

- nodes:

  Optional data frame with one row per node, including nodes without
  edges.

- id_col:

  Identifier column in `nodes`.

- feature_cols:

  Numeric columns of `nodes` used as node features.

- label_col:

  Column of `nodes` used as node labels.

- edge_feature_cols:

  Numeric columns of `edges` used as edge features.

- weight_col:

  Column of `edges` holding edge weights.

- edge_label_col:

  Column of `edges` holding edge labels.

- directed:

  Whether edges are directed.

- g:

  A `graph_data` object.

## Value

A `graph_data` object.

## Examples

``` r
edges <- data.frame(from = c("a", "b", "c"), to = c("b", "c", "a"),
                    w = c(0.5, 1, 2))
g <- graph_from_edgelist(edges, weight_col = "w")
g$node_ids
#> [1] "a" "b" "c"
edgelist_from_graph(g)
#>   from to weight
#> 1    a  b    0.5
#> 2    b  c    1.0
#> 3    c  a    2.0
```
