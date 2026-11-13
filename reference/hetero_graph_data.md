# Heterogeneous graph data

Stores a graph with several node types and several relation types. Each
node type has its own feature matrix (feature sizes may differ), labels
and masks; each relation is identified by an explicit triple
`(source type, relation, target type)` and stores its edges as a 2 by E
matrix whose first row indexes nodes **of the source type** and whose
second row indexes nodes **of the target type**, both one based.

## Usage

``` r
hetero_graph_data(nodes, edges, metadata = list(), validate = TRUE)

validate_hetero_graph(hg)
```

## Arguments

- nodes:

  Named list, one entry per node type: a feature matrix, a node count,
  or a list with any of `x`, `y`, `num_nodes`, `node_ids`, `train_mask`,
  `val_mask`, `test_mask`.

- edges:

  Named list, one entry per relation. Name an entry
  `"src__relation__dst"` and give a 2 by E (or E by 2) matrix, or give a
  list with `edge_index` and optional `src`, `rel`, `dst`, `edge_attr`,
  `edge_weight`, `edge_y`, `edge_time`. `edge_time` is a relative time
  per edge (for example the target's year minus the source's), used by
  the relative temporal encoding of
  [`hgt_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hgt_conv.md);
  reverse relations get its negative.

- metadata:

  Named list of free metadata.

- validate:

  Run `validate_hetero_graph()`.

- hg:

  A `hetero_graph_data` object.

## Value

An object of class `hetero_graph_data`.

## Details

Relations are directed. To let information flow both ways, add reverse
relations with
[`add_reverse_relations()`](https://adamrsophie.github.io/graphlearnR/reference/add_reverse_relations.md);
the pairing between a relation and its reverse is recorded so that link
prediction splits remove held out edges from both
([`split_edges_hetero()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges_hetero.md)).

## Examples

``` r
hg <- hetero_graph_data(
  nodes = list(drug = matrix(rnorm(6), 3, 2), protein = matrix(rnorm(12), 4, 3)),
  edges = list(drug__targets__protein = rbind(c(1, 2, 3), c(1, 1, 4)))
)
hg
#> <hetero_graph_data> 2 node types, 1 relations
#>   node drug                3 nodes, 2 features
#>   node protein             4 nodes, 3 features
#>   edge drug__targets__protein: 3
hg <- add_reverse_relations(hg)
names(hg$edges)
#> [1] "drug__targets__protein"     "protein__rev_targets__drug"
```
