# Graph data: structures, conversion, and batching

``` r

library(graphlearnR)
```

Every model in graphlearnR consumes the same data structure, a
`graph_data` object. This article explains what it stores, the rules it
enforces, and how to get your data into it.

## The data model

A graph with `N` nodes and `E` stored edges has

| field | shape | meaning |
|----|----|----|
| `x` | `N x F` matrix | node features |
| `edge_index` | `2 x E` integer matrix | row 1 source, row 2 target, 1-based |
| `y` | length `N` | node labels (factor or numeric) |
| `edge_attr` | `E x D` matrix | edge features |
| `edge_weight` | length `E` | scalar edge weights |
| `edge_y` | length `E` | edge labels |
| `graph_y` | length 1 | graph label |
| `train_mask`, `val_mask`, `test_mask` | length `N` logical | node splits |
| `node_ids` | length `N` | original identifiers |
| `edge_pair` | length `E` | pair id of each stored edge (undirected graphs) |

``` r

g <- graph_data(
  x = matrix(c(1, 0, 0, 1, 1, 1), nrow = 3, byrow = TRUE),
  edge_index = rbind(c(1, 2), c(2, 3)),
  y = factor(c("a", "b", "a")),
  edge_weight = c(0.5, 2)
)
g
#> <graph_data> directed graph: 3 nodes, 2 edges
#>   node features [3 x 2]
#>   edge weights
#>   node labels: factor with 2 levels
```

### Indexing is 1-based

Node `1` is the first row of `x`. Edge index matrices, seeds for
sampling, node pairs for link prediction, and every returned node id use
the same convention. The only place a 0-based index appears is at an
explicit boundary with other software, through
[`to_zero_based()`](https://adamrsophie.github.io/graphlearnR/reference/to_zero_based.md)
and
[`from_zero_based()`](https://adamrsophie.github.io/graphlearnR/reference/to_zero_based.md).

``` r

to_zero_based(g$edge_index)
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    1    2
```

### Direction

An edge `(i, j)` sends a message from `i` to `j`; layers aggregate over
the incoming edges of each node. A directed graph is stored as given.

For an undirected graph, give each pair once. graphlearnR stores both
directions next to each other and records the pair id. Edge level fields
such as weights are copied to both directions.

``` r

u <- graph_data(edge_index = rbind(c(1, 2), c(2, 3)), edge_weight = c(1, 5),
                directed = FALSE)
u$edge_index
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    2    2    3
#> [2,]    2    1    3    2
u$edge_weight
#> [1] 1 1 5 5
u$edge_pair
#> [1] 1 1 2 2
num_edges(u)
#> [1] 4
num_edges(u, unique_pairs = TRUE)
#> [1] 2
```

Giving both directions of an undirected pair is almost always a mistake,
so it raises a warning.

``` r

w <- graph_data(edge_index = rbind(c(1, 2), c(2, 1)), directed = FALSE)
#> Warning: Undirected input contains reciprocal pairs (both u-v and v-u). Supply
#> each undirected edge once, or use `directed = TRUE` if the list already holds
#> both directions.
```

### Validation

[`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md)
validates its input and fails with a message that names the problem: out
of range or fractional indices, missing values, mismatched lengths, and
overlapping masks.

``` r

try(graph_data(edge_index = rbind(1, 5), num_nodes = 3))
#> Error : Invalid graph_data:
#> * `edge_index` refers to nodes outside 1:3 (first bad edge: column 1, 1 -> 5).
try(graph_data(x = diag(2), train_mask = c(TRUE, FALSE), test_mask = c(TRUE, FALSE)))
#> Error : Invalid graph_data:
#> * `train_mask` and `test_mask` overlap.
```

[`validate_graph()`](https://adamrsophie.github.io/graphlearnR/reference/validate_graph.md)
runs the same checks on an object you have modified by hand.

### Inspecting a graph

``` r

k <- dataset_karate()
c(nodes = num_nodes(k), edges = num_edges(k), features = num_node_features(k))
#>    nodes    edges features 
#>       34      156       34
has_self_loops(k)
#> [1] FALSE
has_isolated_nodes(k)
#> [1] FALSE
head(degree(k))
#> [1] 16  9 10  6  3  4
```

## Conversion

### From and to igraph

[`from_igraph()`](https://adamrsophie.github.io/graphlearnR/reference/from_igraph.md)
reads vertex names into `node_ids`, the `weight` edge attribute into
`edge_weight`, and any attributes you name into features.

``` r

ig <- igraph::make_graph("Zachary")
igraph::V(ig)$age <- seq_len(igraph::vcount(ig))
gi <- from_igraph(ig, x_attrs = "age")
gi
#> <graph_data> undirected graph: 34 nodes, 156 edges (78 undirected pairs)
#>   node features [34 x 1]
back <- to_igraph(gi)
igraph::ecount(back)
#> [1] 78
```

An undirected igraph object becomes an undirected `graph_data`; the
round trip keeps the edge count.

### From an edge list

Real data rarely numbers nodes `1:N`.
[`graph_from_edgelist()`](https://adamrsophie.github.io/graphlearnR/reference/graph_from_edgelist.md)
maps arbitrary identifiers through an optional node table, which also
supplies features and labels. Nodes that appear in the table but in no
edge are kept as isolated nodes.

``` r

edges <- data.frame(from = c("ann", "bob", "cat"), to = c("bob", "cat", "ann"),
                    strength = c(3, 1, 2))
nodes <- data.frame(id = c("ann", "bob", "cat", "dan"), age = c(31, 45, 27, 52),
                    group = c("x", "y", "x", "y"))
ge <- graph_from_edgelist(edges, nodes = nodes, feature_cols = "age",
                          label_col = "group", weight_col = "strength")
ge
#> <graph_data> directed graph: 4 nodes, 3 edges
#>   node features [4 x 1]
#>   edge weights
#>   node labels: factor with 2 levels
ge$node_ids
#> [1] "ann" "bob" "cat" "dan"
edgelist_from_graph(ge)
#>   from  to weight
#> 1  ann bob      3
#> 2  bob cat      1
#> 3  cat ann      2
```

### From an adjacency matrix

Dense and sparse (`Matrix`) adjacency matrices both work. A symmetric
matrix gives an undirected graph unless `directed = TRUE`. Entry
`A[i, j]` is the edge from `i` to `j`.

``` r

A <- Matrix::sparseMatrix(i = c(1, 2, 3), j = c(2, 3, 1), x = c(1, 2, 3), dims = c(3, 3))
ga <- graph_from_adjacency(A)
ga$edge_index
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    2    3    1
adjacency_from_graph(ga)
#> 3 x 3 sparse Matrix of class "dgCMatrix"
#>           
#> [1,] . 1 .
#> [2,] . . 2
#> [3,] 3 . .
```

### To tensors

[`graph_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/graph_tensors.md)
converts every field to a torch tensor on a chosen device: features to
floating point, indices and class labels to `long`, and masks to `bool`.
Indices stay 1-based, which is what R torch indexing expects.

``` r

tt <- graph_tensors(k)
tt
#> <graph_tensors> 34 nodes, 156 edges on cpu
tt$edge_index$dtype
#> torch_Long
```

## Many graphs: batching

Graph level tasks train on many small graphs.
[`graph_batch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
joins them into one disconnected graph: node indices are offset so no
edge crosses between graphs, and `batch` records which graph each node
belongs to.

``` r

gs <- dataset_synthetic_graphs(3, seed = 1)
b <- graph_batch(gs)
b
#> <graph_batch> 3 graphs, 25 nodes, 64 edges
b$node_ptr
#> [1]  0  5 14 25
table(b$batch)
#> 
#>  1  2  3 
#>  5  9 11
```

[`unbatch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
reverses the operation exactly.

``` r

parts <- unbatch(b)
identical(parts[[2]]$edge_index, gs[[2]]$edge_index)
#> [1] TRUE
```

## Datasets

[`graph_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md)
wraps a list of graphs and applies an optional transform when a graph is
accessed.
[`write_graph_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md)
and
[`graph_dataset_disk()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md)
keep large collections on disk, one file per graph, loading each on
demand.

``` r

ds <- graph_dataset(dataset_synthetic_graphs(20, seed = 2),
                    transform = add_degree_features)
ds
#> <graph_dataset> 20 graphs
ncol(ds[[1]]$x)
#> [1] 2
```

Public benchmark datasets are downloaded, not bundled, and their MD5
checksum is verified before use. `download_tu_dataset("MUTAG")` reads a
TU Dortmund graph classification dataset and
`download_planetoid_raw("cora")` reads the raw LINQS release of Cora;
read their licenses before redistributing them.

## Transforms

Transforms are plain functions from a graph to a graph, so they compose
with
[`compose_transforms()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
or with the native pipe.

``` r

g2 <- graph_data(x = diag(3), edge_index = rbind(c(1, 1, 2), c(2, 2, 3)))
g2 |> coalesce_edges() |> add_self_loops() |> (\(h) h$edge_index)()
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    2    1    2    3
#> [2,]    2    3    1    2    3
ks <- split_nodes(dataset_karate("degree"), seed = 1)
head(normalize_features(ks, stats_mask = ks$train_mask)$x)
#>             [,1]
#> [1,]  2.19650230
#> [2,]  0.69645195
#> [3,]  0.91074486
#> [4,]  0.05357323
#> [5,] -0.58930550
#> [6,] -0.37501259
```

Pass `stats_mask` to
[`normalize_features()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
to compute centering and scaling statistics from training nodes only, so
test information does not leak into the features.
