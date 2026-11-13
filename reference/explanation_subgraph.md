# Explanation subgraph

Turns an edge attribution into a subgraph explanation: the `k` edges
with the largest absolute attribution (and, for undirected graphs, their
reverse directions), the nodes they touch, and the target node. The
result is a `graph_data` object with local 1-based indices; its
`node_ids` hold the original node identifiers (`g$node_ids`, which are
the node indices unless the graph has external identifiers) so the
subgraph can be mapped back, and `edge_original` the original edge
positions.

## Usage

``` r
explanation_subgraph(g, explanation, k = 5L)
```

## Arguments

- g:

  The `graph_data` object that was explained.

- explanation:

  A `graph_explanation` with an `edges` table, for example from
  [`explain_edges()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
  or
  [`gnn_explainer()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md).

- k:

  Number of top edges (directions of one undirected pair count once).

## Value

A `graph_data` object.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
                           "node_classification", epochs = 20, seed = 1)
  sub <- explanation_subgraph(g, explain_edges(fit, g, target = 1), k = 5)
  sub$node_ids
}
#> [1]  1  5 11 12 22 32
```
