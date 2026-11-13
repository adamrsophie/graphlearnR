# Graph transformers and structural encodings

``` r

library(graphlearnR)
```

Message passing layers only look at neighbors, so information needs many
layers to travel far, and some structures are invisible to them. Graph
transformers let every node attend to every other node in its graph and
inject structure through *encodings*: extra node features or attention
biases computed from the graph.

## Positional and structural encodings

### Laplacian eigenvectors

[`laplacian_pe()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
returns the `k` eigenvectors of the normalized Laplacian L = I -
D^{-1/2} A D^{-1/2} with the smallest nonzero eigenvalues. They are
smooth coordinates on the graph: nearby nodes get similar values.

``` r

g <- dataset_karate(features = "none")
pe <- laplacian_pe(g, k = 4)
dim(pe)
#> [1] 34  4
round(crossprod(pe), 6)
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    0
#> [2,]    0    1    0    0
#> [3,]    0    0    1    0
#> [4,]    0    0    0    1
```

Eigenvectors are defined only up to sign, so a model should not depend
on the sign. Flip signs at random during training with
[`random_sign_flip()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md).

``` r

flipped <- random_sign_flip(pe, seed = 1)
all.equal(abs(flipped), abs(pe))
#> [1] TRUE
```

Graphs with fewer than `k + 1` nodes are padded with zero columns, so
every graph in a dataset gets the same width.

### Random walk encodings

[`random_walk_pe()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
returns, for each node, the probability that a random walk returns to it
after 1, \dots, k steps. Unlike eigenvectors these are unique and
permutation equivariant, and they reveal cycles: a node on a triangle
has a positive return probability at step 3.

``` r

rw <- random_walk_pe(g, k = 4)
head(round(rw, 3))
#>      [,1]  [,2]  [,3]  [,4]
#> [1,]    0 0.325 0.101 0.201
#> [2,]    0 0.262 0.059 0.130
#> [3,]    0 0.216 0.044 0.105
#> [4,]    0 0.204 0.060 0.087
#> [5,]    0 0.215 0.024 0.105
#> [6,]    0 0.286 0.081 0.154
```

[`add_positional_encoding()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
appends either encoding to the node features.

``` r

g_pe <- add_positional_encoding(dataset_karate("degree"), "random_walk", k = 6)
ncol(g_pe$x)
#> [1] 7
```

### Shortest path distances

[`shortest_path_distances()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
gives hop distances, capped at `max_dist`. Unreachable pairs get
`max_dist + 1`, a separate bucket.

``` r

ring <- graph_data(edge_index = rbind(1:6, c(2:6, 1)), directed = FALSE)
shortest_path_distances(ring)
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    0    1    2    3    2    1
#> [2,]    1    0    1    2    3    2
#> [3,]    2    1    0    1    2    3
#> [4,]    3    2    1    0    1    2
#> [5,]    2    3    2    1    0    1
#> [6,]    1    2    3    2    1    0
```

## The Graphormer style model

[`graph_transformer_model()`](https://adamrsophie.github.io/graphlearnR/reference/graph_transformer_model.md)
follows Graphormer (Ying et al. 2021):

- a *centrality encoding* adds a learned embedding of each node’s degree
  to its input;
- each attention head adds a learned *spatial bias* b\_{\phi(i,j)} that
  depends on the shortest path distance between nodes i and j;
- attention is computed separately inside each graph of a batch, using a
  dense padded layout with a mask, so nodes never attend across graphs.

``` r

library(torch)
graphs <- dataset_synthetic_graphs(80, seed = 1)
labels <- vapply(graphs, function(h) as.character(h$graph_y), "")
sp <- split_graphs(80, labels = labels, seed = 1)
torch_manual_seed(1)
gt <- graph_transformer_model(input_dim = 1, hidden_dim = 32, output_dim = 2,
                              num_layers = 2, heads = 4, max_dist = 5, dropout = 0)
fit <- train_graph_model(gt, graphs[sp$train], "graph_classification",
                         val_data = graphs[sp$val], epochs = 30, batch_size = 16,
                         seed = 1)
evaluate_graph_model(fit, graphs[sp$test])
#>              metric        value
#> 1          accuracy 1.000000e+00
#> 2 balanced_accuracy 1.000000e+00
#> 3   precision_macro 1.000000e+00
#> 4      recall_macro 1.000000e+00
#> 5          f1_macro 1.000000e+00
#> 6               ece 1.609325e-06
#> 7             brier 5.983525e-12
#> 8           roc_auc 1.000000e+00
#> 9 average_precision 1.000000e+00
```

The constant node features of these graphs carry no information; the
model separates cycles from wheels through the degree and distance
encodings alone.

### Dense batching

Full attention needs the nodes of each graph side by side.
[`to_dense_batch()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_batch.md)
converts node features and a `batch` vector into a padded graphs by
nodes by features tensor and a mask;
[`from_dense_batch()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_batch.md)
undoes it.

``` r

b <- graph_batch(graphs[1:3])
d <- to_dense_batch(torch_tensor(b$x), b$batch)
dim(d$x)
#> [1]  3 11  1
as.array(d$mask)
#>      [,1] [,2] [,3] [,4] [,5] [,6]  [,7]  [,8]  [,9] [,10] [,11]
#> [1,] TRUE TRUE TRUE TRUE TRUE TRUE FALSE FALSE FALSE FALSE FALSE
#> [2,] TRUE TRUE TRUE TRUE TRUE TRUE  TRUE  TRUE  TRUE FALSE FALSE
#> [3,] TRUE TRUE TRUE TRUE TRUE TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
```

Memory grows with the square of the largest graph, so this design suits
molecule sized graphs. For large single graphs, use message passing
layers, or
[`transformer_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md),
which attends over neighbors only.

### Attention maps

The layers keep their attention weights, with shape graphs by heads by
nodes by nodes. Padding positions get zero weight.

``` r

gt$eval()
invisible(graphlearnR:::model_forward(gt, graph_tensors(b)))
att <- gt$layers[[1]]$last_attention
dim(att)
#> [1]  3  4 11 11
```

## Choosing an architecture

- Start with message passing
  ([`gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/gnn_model.md));
  it is cheaper and often as accurate.
- Add random walk or Laplacian encodings to a message passing model when
  structure matters more than features; it is a one line change.
- Use the graph transformer for graph level tasks on small graphs where
  long range interactions matter.
