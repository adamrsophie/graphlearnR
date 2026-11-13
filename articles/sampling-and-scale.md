# Neighbor sampling and mini-batch training

``` r

library(graphlearnR)
```

Full batch training runs every layer on every node in every step. Memory
then grows with the whole graph. A model with L layers only needs the L
hop neighborhood of the nodes in the loss, so large graphs are trained
on sampled subgraphs around small batches of *seed* nodes, as in
GraphSAGE (Hamilton, Ying, and Leskovec 2017).

## Sampling a computation graph

[`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md)
starts at the seeds and, hop by hop, samples at most `fanouts[h]`
incoming edges of each frontier node. The result is an ordinary
`graph_data` with extra fields.

``` r

g <- dataset_karate()
sg <- sample_neighbors(g, seeds = c(1, 34), fanouts = c(3, 2), seed = 1)
sg
#> <graph_data> directed graph: 13 nodes, 17 edges
#>   node features [13 x 34]
#>   node labels: factor with 2 levels
sg$n_id
#>  [1]  1 34  6 12 14 19 21 27 11  2  4 33 30
sg$hop
#>  [1] 0 0 1 1 1 1 1 1 2 2 2 2 2
sg$num_seeds
#> [1] 2
```

- The seeds come first, in the order given, so `out[1:num_seeds, ]` are
  the predictions that enter the loss.
- `n_id` maps local node `i` back to the original node `n_id[i]`, and
  `e_id` does the same for edges.
- `hop` is the hop at which each node was first reached.

Mapping local edges back to original ids recovers edges of the input
graph:

``` r

local <- sg$edge_index[, 1:4]
rbind(from = sg$n_id[local[1, ]], to = sg$n_id[local[2, ]])
#>      [,1] [,2] [,3] [,4]
#> from    6   12   14   19
#> to      1    1    1   34
```

A fanout of `-1` keeps every neighbor. With full fanouts the sample is
the exact L hop computation graph, so a model gives the same predictions
on the seeds as on the full graph.
[`k_hop_subgraph()`](https://adamrsophie.github.io/graphlearnR/reference/k_hop_subgraph.md)
is the shortcut for that case.

``` r

library(torch)
torch_manual_seed(1)
m <- gnn_model("sage", 34, 16, 2, num_layers = 2)
m$eval()
full <- as.array(graphlearnR:::model_forward(m, graph_tensors(g)))
sub <- k_hop_subgraph(g, c(1, 34), k = 2)
part <- as.array(graphlearnR:::model_forward(m, graph_tensors(sub)))
max(abs(part[1:2, ] - full[c(1, 34), ]))
#> [1] 0
```

## Loaders

[`node_loader()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md)
splits the seed nodes into batches and samples a subgraph for each. Each
epoch uses a different, reproducible shuffle and sample.

``` r

g <- split_nodes(g, seed = 1)
ld <- node_loader(g, batch_size = 8, fanouts = c(5, 5), seed = 1)
ld
#> <node_loader> 20 seed nodes in 3 batches of up to 8, fanouts 5, 5
batches <- loader_batches(ld, epoch = 1)
vapply(batches, function(b) c(seeds = b$num_seeds, nodes = num_nodes(b)), numeric(2))
#>       [,1] [,2] [,3]
#> seeds    8    8    4
#> nodes   26   30   26
```

[`graph_loader()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md)
does the same for graph level datasets, yielding `graph_batch` objects.

## Mini-batch training

Give
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
the `fanouts` and `batch_size` arguments and it trains on sampled
subgraphs. The loss is computed on seed nodes only; sampled neighbors
provide context but no gradient signal of their own.

``` r

big <- split_nodes(dataset_sbm(c(300, 300, 300), p_in = 0.03, p_out = 0.002,
                               feature_signal = 0.5, seed = 2), seed = 2)
torch_manual_seed(2)
fit <- train_graph_model(gnn_model("sage", 8, 32, 3), big, "node_classification",
                         epochs = 15, fanouts = c(10, 5), batch_size = 128,
                         seed = 2)
fit
#> <graph_fit> node_classification, 15 epochs on cpu
#>   final train loss 0.0594, val loss 0.0169, val accuracy 0.9944
#>   best epoch 13
evaluate_graph_model(fit, big)
#>              metric       value
#> 1          accuracy 0.994444444
#> 2 balanced_accuracy 0.994444444
#> 3   precision_macro 0.994535519
#> 4      recall_macro 0.994444444
#> 5          f1_macro 0.994444059
#> 6               ece 0.006840451
#> 7             brier 0.012466965
#> 8           roc_auc 0.997037037
```

Prediction and evaluation run on the full graph, so the reported numbers
do not depend on the sampling.

## Negative sampling

Link prediction needs pairs that are not edges.
[`negative_sampling()`](https://adamrsophie.github.io/graphlearnR/reference/negative_sampling.md)
draws them by rejection, excluding the given positives, self loops
(unless allowed), and duplicates. For undirected graphs it returns
canonical pairs with `from < to` and excludes both directions of each
positive. Bipartite sampling draws sources and targets from different
node sets.

``` r

pos <- dataset_karate()$edge_index
neg <- negative_sampling(pos, num_nodes = 34, num_samples = 5, directed = FALSE,
                         seed = 1)
neg
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    4    6   10   14   18
#> [2,]   33   34   23   20   28
negative_sampling(rbind(1:3, 1:3), num_src = 5, num_dst = 4, num_samples = 4,
                  seed = 1)
#>      [,1] [,2] [,3] [,4]
#> [1,]    4    5    3    3
#> [2,]    1    1    1    2
```

## Practical notes

- Throughput depends mostly on the number of sampled edges, which is
  about `batch_size * prod(fanouts)`. Start with fanouts like
  `c(10, 10)` and lower them if memory is tight.
- Sampling runs in R on the CPU; the model runs on whichever device you
  pass as `device`. For very large graphs, keep the graph in memory once
  and let the loader slice it, rather than building many graph copies.
- The `benchmarks/` folder of the source repository has timing scripts
  for layers and for full batch against sampled training. Their results
  are produced on GitHub Actions runners and reported with the hardware
  they ran on.
