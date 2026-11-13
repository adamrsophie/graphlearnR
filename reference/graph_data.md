# Create a graph data object

`graph_data()` stores one homogeneous graph: node features, an edge
index, optional edge features and weights, labels at node, edge and
graph level, node and edge type codes, split masks and free metadata.
Data are kept as ordinary R vectors and matrices;
[`graph_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/graph_tensors.md)
turns them into `torch` tensors once, on the device you choose.

## Usage

``` r
graph_data(
  x = NULL,
  edge_index = NULL,
  y = NULL,
  edge_attr = NULL,
  edge_weight = NULL,
  edge_y = NULL,
  graph_y = NULL,
  graph_attr = NULL,
  num_nodes = NULL,
  directed = TRUE,
  node_ids = NULL,
  node_type = NULL,
  edge_type = NULL,
  train_mask = NULL,
  val_mask = NULL,
  test_mask = NULL,
  metadata = list(),
  validate = TRUE
)
```

## Arguments

- x:

  Node features: numeric matrix with one row per node, a numeric vector
  (one feature), a numeric data frame, or `NULL`.

- edge_index:

  Edges as a 2 by E matrix (row 1 source, row 2 target), an E by 2
  matrix, or a data frame with columns `from` and `to`. Values are one
  based node indices. `NULL` means no edges.

- y:

  Node labels: factor (classification), integer or numeric vector, or
  numeric matrix (multi target), one entry per node. Character labels
  are converted to a factor (as are character `edge_y` and `graph_y`).

- edge_attr:

  Edge features, one row per input edge.

- edge_weight:

  Positive or signed numeric edge weights, one per input edge.

- edge_y:

  Edge labels, one per input edge.

- graph_y:

  Graph label (scalar, factor of length one, or numeric vector).

- graph_attr:

  Graph level features (numeric vector).

- num_nodes:

  Number of nodes. Inferred from `x`, `y`, `node_ids` or the largest
  edge index when `NULL`.

- directed:

  Whether edges are directed. See the edge direction section.

- node_ids:

  External node identifiers, one per node. Default `seq_len(num_nodes)`.

- node_type, edge_type:

  Optional integer or factor type codes per node and per input edge.
  They are stored, not interpreted; typed learning uses
  [`hetero_graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_graph_data.md).

- train_mask, val_mask, test_mask:

  Logical node masks.

- metadata:

  Named list of free metadata (source, version, notes).

- validate:

  Run
  [`validate_graph()`](https://adamrsophie.github.io/graphlearnR/reference/validate_graph.md)
  after construction.

## Value

An object of class `graph_data`.

## Indexing convention

Node indices are **one based** everywhere in the public API: node `i` is
row `i` of `x`, and `edge_index` holds values in `1:num_nodes`. R
`torch` index operations (`index_select`, `index_add`, `scatter_add`,
`scatter_reduce`) are also one based, so the stored indices are passed
to `torch` unchanged. The only conversion to zero based indices happens
in
[`to_zero_based()`](https://adamrsophie.github.io/graphlearnR/reference/to_zero_based.md),
used when exporting to formats that require it.

## Edge direction

`edge_index` is a 2 by E matrix; column `j` is a directed edge from
`edge_index[1, j]` (source) to `edge_index[2, j]` (target). Messages
flow from source to target.

With `directed = FALSE`, supply each undirected edge **once**. The
stored graph holds both directions, `(u, v)` and `(v, u)`, next to each
other; a self loop is stored once. `edge_pair` maps every stored
directed edge to the input edge it came from, and edge level values
(`edge_attr`, `edge_weight`, `edge_y`, `edge_type`) are copied to both
directions. If the input already contains reciprocal pairs, a warning is
given, because those pairs would then be stored twice in each direction.

## Examples

``` r
g <- graph_data(
  x = matrix(rnorm(8), 4, 2),
  edge_index = rbind(c(1, 2, 3), c(2, 3, 4)),
  y = factor(c("a", "b", "a", "b")),
  directed = FALSE
)
g
#> <graph_data> undirected graph: 4 nodes, 6 edges (3 undirected pairs)
#>   node features [4 x 2]
#>   node labels: factor with 2 levels
num_nodes(g)
#> [1] 4
num_edges(g)
#> [1] 6
num_edges(g, unique_pairs = TRUE)
#> [1] 3
```
