# Batch graphs into one disconnected graph

Combines several graphs into a single graph whose components are the
input graphs. Node indices of graph `k` are shifted by the number of
nodes in graphs `1, ..., k - 1`, so no edge ever joins two input graphs.
The result records:

## Usage

``` r
graph_batch(graphs)

unbatch(batch)
```

## Arguments

- graphs:

  A list of `graph_data` objects with the same number of node features,
  edge features and the same direction.

- batch:

  A `graph_batch` object, or a heterogeneous batch from
  [`hetero_batch()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_batch.md).

## Value

An object of class `c("graph_batch", "graph_data")`.

## Details

- `batch`: graph membership of every node (`1:num_graphs`),

- `node_ptr`, `edge_ptr`: offsets, so the nodes of graph `k` are
  `(node_ptr[k] + 1):node_ptr[k + 1]`, and likewise for edges,

- `original_node_ids`: the `node_ids` of each node in its own graph,

- `graph_y` and `graph_attr` stacked one row (or element) per graph.

A `graph_batch` is also a `graph_data`, so every layer and pooling
function accepts it. `unbatch()` reconstructs the input graphs.

## Examples

``` r
g1 <- graph_data(x = matrix(1, 2, 1), edge_index = rbind(1, 2), graph_y = 0)
g2 <- graph_data(x = matrix(2, 3, 1), edge_index = rbind(c(1, 2), c(2, 3)),
                 graph_y = 1)
b <- graph_batch(list(g1, g2))
b$num_graphs
#> [1] 2
b$edge_index
#>      [,1] [,2] [,3]
#> [1,]    1    3    4
#> [2,]    2    4    5
b$batch
#> [1] 1 1 2 2 2
length(unbatch(b))
#> [1] 2
```
