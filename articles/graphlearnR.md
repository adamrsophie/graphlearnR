# Getting started with graphlearnR

graphlearnR is a framework for machine learning on graphs written in R
on top of the [torch](https://torch.mlverse.org) package. Nothing in it
calls Python. This article walks through one complete workflow: load a
graph, split it, train a graph neural network, evaluate it, and predict.
Later articles cover each step in depth.

``` r

library(graphlearnR)
library(torch)
```

LibTorch must be installed once with
[`torch::install_torch()`](https://torch.mlverse.org/docs/reference/install_torch.html).
Every code chunk in these articles runs only when it is available.

## A first graph

Zachary’s karate club is a social network of 34 club members. The club
split into two factions, which is the node label.
[`dataset_karate()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_karate.md)
returns it as a `graph_data` object with one hot node identity features.

``` r

g <- dataset_karate()
g
#> <graph_data> undirected graph: 34 nodes, 156 edges (78 undirected pairs)
#>   node features [34 x 34]
#>   node labels: factor with 2 levels
summary(g)
#> Graph summary: 34 nodes, 156 edges (undirected)
#>   node features: 34, edge features: 0
#>   self loops: FALSE, duplicate edges: FALSE, isolated nodes: 0
#>   in degree:
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>   1.000   2.000   3.000   4.588   5.000  17.000 
#>   node label counts:
#> 
#>  Mr. Hi Officer 
#>      17      17
```

The graph is undirected. graphlearnR stores each undirected pair in both
directions, so message passing sees both, and records which stored edges
belong to the same pair in `g$edge_pair`. Node indices are 1-based
everywhere, like the rest of R.

``` r

g$edge_index[, 1:6]
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    1    2    1    3    1    4
#> [2,]    2    1    3    1    4    1
g$edge_pair[1:6]
#> [1] 1 1 2 2 3 3
```

## Split the nodes

[`split_nodes()`](https://adamrsophie.github.io/graphlearnR/reference/split_nodes.md)
adds disjoint logical masks. With a factor label it stratifies by class,
so each split keeps the class balance.

``` r

g <- split_nodes(g, train = 0.5, val = 0.25, test = 0.25, seed = 42)
table(train = g$train_mask, label = g$y)
#>        label
#> train   Mr. Hi Officer
#>   FALSE      8       8
#>   TRUE       9       9
```

## Build and train a model

[`gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/gnn_model.md)
builds a stack of message passing layers. Here a two layer GraphSAGE
network maps the 34 input features to 16 hidden units and then to two
class scores.

``` r

torch_manual_seed(1)
model <- gnn_model("sage", input_dim = 34, hidden_dim = 16, output_dim = 2,
                   dropout = 0.3)
model
#> An `nn_module` containing 1,666 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • convs: <nn_module_list> #1,632 parameters
#> • head: <nn_linear> #34 parameters
```

[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
runs the training loop. The loss for node classification is cross
entropy on the training nodes only; the validation nodes drive early
stopping, and the best weights are restored at the end.

``` r

fit <- train_graph_model(model, g, task = "node_classification",
                         epochs = 150, lr = 0.01, weight_decay = 5e-4,
                         early_stopping = 20, seed = 1)
fit
#> <graph_fit> node_classification, 45 epochs on cpu
#>   final train loss 0.0022, val loss 0.4130, val accuracy 0.8750
#>   best epoch 25
tail(fit$history)
#>    epoch   train_loss  val_loss val_metric   lr seconds
#> 40    40 0.0028915016 0.3628182      0.875 0.01   0.010
#> 41    41 0.0004772095 0.3762871      0.875 0.01   0.010
#> 42    42 0.0017317395 0.3893111      0.875 0.01   0.016
#> 43    43 0.0035787998 0.3998920      0.875 0.01   0.010
#> 44    44 0.0023367275 0.4064358      0.875 0.01   0.010
#> 45    45 0.0022041511 0.4130065      0.875 0.01   0.009
```

## Evaluate and predict

[`evaluate_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_graph_model.md)
scores the held out test nodes.

``` r

evaluate_graph_model(fit, g)
#>              metric      value
#> 1          accuracy 1.00000000
#> 2 balanced_accuracy 1.00000000
#> 3   precision_macro 1.00000000
#> 4      recall_macro 1.00000000
#> 5          f1_macro 1.00000000
#> 6               ece 0.06625122
#> 7             brier 0.01589126
#> 8           roc_auc 1.00000000
```

[`predict()`](https://rdrr.io/r/stats/predict.html) returns a data frame
with one row per node, a predicted class, and a probability per class.
Other types return only the classes, the probability matrix, the raw
scores, or the node embeddings.

``` r

pred <- predict(fit, g)
head(pred)
#>   .row node_id .pred_class .prob_Mr. Hi .prob_Officer
#> 1    1       1      Mr. Hi    0.9587319    0.04126804
#> 2    2       2      Mr. Hi    0.9602939    0.03970613
#> 3    3       3      Mr. Hi    0.8446112    0.15538870
#> 4    4       4      Mr. Hi    0.9614742    0.03852574
#> 5    5       5      Mr. Hi    0.9683224    0.03167759
#> 6    6       6      Mr. Hi    0.9551980    0.04480197
emb <- predict(fit, g, type = "embedding")
dim(emb)
#> [1] 34 16
```

## Where to go next

- [Graph
  data](https://adamrsophie.github.io/graphlearnR/articles/graph-data.md):
  the data model, conversions from igraph, edge lists and sparse
  matrices, batching, and transforms.
- [Message
  passing](https://adamrsophie.github.io/graphlearnR/articles/message-passing.md):
  scatter aggregation, the built in layers, and how to write a custom
  layer.
- [Node, graph, and link
  tasks](https://adamrsophie.github.io/graphlearnR/articles/node-graph-link-tasks.md):
  every task the training loop supports.
- [Sampling and
  scale](https://adamrsophie.github.io/graphlearnR/articles/sampling-and-scale.md):
  neighbor sampling and mini-batch training.
- [Heterogeneous
  graphs](https://adamrsophie.github.io/graphlearnR/articles/heterogeneous-graphs.md),
  [temporal
  graphs](https://adamrsophie.github.io/graphlearnR/articles/temporal-graphs.md),
  and [graph
  transformers](https://adamrsophie.github.io/graphlearnR/articles/graph-transformers.md).
- [Explainability](https://adamrsophie.github.io/graphlearnR/articles/explainability.md)
  and [evaluation and
  reproducibility](https://adamrsophie.github.io/graphlearnR/articles/evaluation-and-reproducibility.md).
