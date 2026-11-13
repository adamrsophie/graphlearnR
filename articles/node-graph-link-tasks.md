# Node, graph, link, and edge tasks

``` r

library(graphlearnR)
library(torch)
```

[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
handles seven tasks with one interface. The task decides which rows
enter the loss, which loss is the default, how validation works, and
what [`predict()`](https://rdrr.io/r/stats/predict.html) and
[`evaluate_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_graph_model.md)
return.

| task | data | loss rows | default loss | metrics |
|----|----|----|----|----|
| `node_classification` | one graph with masks | training nodes | cross entropy | accuracy, F1, AUC, calibration |
| `node_regression` | one graph with masks | training nodes | MSE | MSE, RMSE, MAE, R^2 |
| `graph_classification` | list of graphs | every graph | cross entropy | as above |
| `graph_regression` | list of graphs | every graph | MSE | as above |
| `link_prediction` | a `link_split` | training pairs and fresh negatives | binary cross entropy | AUC, AP, MRR, <Hits@k> |
| `edge_classification` | one graph with edge masks | training edges | cross entropy | as above |
| `edge_regression` | one graph with edge masks | training edges | MSE | as above |

## Node classification

A stochastic block model has communities with dense links inside and
sparse links between them.
[`dataset_sbm()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_sbm.md)
adds node features that carry a weak class signal, so a good model must
combine features and structure.

``` r

g <- dataset_sbm(c(60, 60, 60), p_in = 0.1, p_out = 0.01, feature_signal = 0.5,
                 seed = 1)
g <- split_nodes(g, seed = 1)
g
#> <graph_data> undirected graph: 180 nodes, 1286 edges (643 undirected pairs)
#>   node features [180 x 8]
#>   node labels: factor with 3 levels
#>   masks: train=108, val=36, test=36
```

Compare a structure blind baseline (a multilayer perceptron on the
features) with three graph models.

``` r

torch_manual_seed(1)
mlp <- nn_module(
  initialize = function() self$net <- graph_mlp(c(8, 32, 3), dropout = 0.3),
  forward = function(x, edge_index, ...) self$net(x)
)
fits <- list(mlp = train_graph_model(mlp(), g, "node_classification", epochs = 100,
                                     seed = 1))
for (a in c("sage", "gcn", "gat")) {
  torch_manual_seed(1)
  fits[[a]] <- train_graph_model(gnn_model(a, 8, 32, 3, heads = 2, dropout = 0.3), g,
                                 "node_classification", epochs = 100, seed = 1)
}
sapply(fits, function(f) {
  ev <- evaluate_graph_model(f, g)
  ev$value[ev$metric == "accuracy"]
})
#>       mlp      sage       gcn       gat 
#> 0.5277778 0.8888889 0.9444444 0.9722222
```

Any `nn_module` whose `forward()` accepts `(x, edge_index, ...)` can be
trained, which is how the baseline above plugs in.

### Regression

Numeric labels select regression. The model output has one column.

``` r

gr <- g
gr$y <- as.integer(g$y) + rnorm(180, sd = 0.2)
torch_manual_seed(2)
fit_r <- train_graph_model(gnn_model("sage", 8, 32, 1), gr, "node_regression",
                           epochs = 100, seed = 1)
evaluate_graph_model(fit_r, gr)
#>   metric     value
#> 1    mse 0.1869997
#> 2   rmse 0.4324347
#> 3    mae 0.3472295
#> 4     r2 0.7380613
head(predict(fit_r, gr))
#>   .row node_id    .pred
#> 1    1       1 1.284679
#> 2    2       2 1.486635
#> 3    3       3 1.304096
#> 4    4       4 1.251207
#> 5    5       5 1.615181
#> 6    6       6 1.114796
```

## Graph classification

[`dataset_synthetic_graphs()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_synthetic_graphs.md)
generates cycles and wheels. Both have the same constant node features,
so only structure separates them. GIN with sum pooling is as powerful as
the Weisfeiler and Lehman test at telling such graphs apart.

``` r

graphs <- dataset_synthetic_graphs(120, seed = 3)
labels <- vapply(graphs, function(h) as.character(h$graph_y), "")
sp <- split_graphs(length(graphs), labels = labels, seed = 3)
lengths(sp)
#> train   val  test 
#>    96    12    12

torch_manual_seed(3)
model <- gnn_model("gin", input_dim = 1, hidden_dim = 32, output_dim = 2,
                   num_layers = 3, pool = "sum", dropout = 0)
fit_g <- train_graph_model(model, graphs[sp$train], "graph_classification",
                           val_data = graphs[sp$val], epochs = 60, batch_size = 16,
                           seed = 3)
evaluate_graph_model(fit_g, graphs[sp$test])
#>              metric        value
#> 1          accuracy 1.000000e+00
#> 2 balanced_accuracy 1.000000e+00
#> 3   precision_macro 1.000000e+00
#> 4      recall_macro 1.000000e+00
#> 5          f1_macro 1.000000e+00
#> 6               ece 0.000000e+00
#> 7             brier 5.953295e-16
#> 8           roc_auc 1.000000e+00
head(predict(fit_g, graphs[sp$test]))
#>   .row graph .pred_class .prob_cycle  .prob_wheel
#> 1    1     1       cycle           1 7.298568e-12
#> 2    2     2       cycle           1 1.078535e-08
#> 3    3     3       cycle           1 6.407741e-13
#> 4    4     4       cycle           1 5.625655e-14
#> 5    5     5       cycle           1 7.298568e-12
#> 6    6     6       cycle           1 5.625655e-14
```

The `pool` argument turns any architecture into a graph level model:
`"sum"`, `"mean"`, `"max"`, or `"attention"`.

## Link prediction

Link prediction asks which missing pairs are likely edges. The
evaluation is only honest if held out edges are invisible during
training, so
[`split_edges()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges.md)
does three things:

1.  It splits each undirected pair, never one direction of it, into
    training, validation, and test sets.
2.  It returns `train_graph`, the graph of training edges only, which is
    the only graph message passing ever sees during training.
3.  It samples validation and test negatives that are not edges anywhere
    in the full graph.

``` r

gl <- dataset_sbm(c(50, 50), p_in = 0.2, p_out = 0.01, seed = 4)
ls <- split_edges(gl, val = 0.1, test = 0.2, seed = 4)
ls
#> <link_split> 100 nodes, undirected pairs: train 372, val 53 (+53 neg), test 106 (+106 neg)
```

The encoder produces node embeddings and a decoder scores pairs. Fresh
training negatives are drawn every epoch.

``` r

torch_manual_seed(4)
enc <- gnn_model("gcn", 8, 32, output_dim = NULL, dropout = 0)
fit_l <- train_graph_model(enc, ls, "link_prediction", epochs = 100, lr = 0.01,
                           seed = 4)
evaluate_graph_model(fit_l)
#>              metric       value
#> 1           roc_auc 0.772516910
#> 2 average_precision 0.694723501
#> 3               mrr 0.081566876
#> 4            hits@1 0.009433962
#> 5           hits@10 0.235849057
#> 6           hits@50 0.981132075
```

Score any pairs with
[`predict()`](https://rdrr.io/r/stats/predict.html). Message passing
uses the training graph.

``` r

predict(fit_l, pairs = rbind(c(1, 1, 60), c(2, 70, 61)), type = "prob")
#> [1] 0.6104220 0.5000000 0.5801775
```

Other decoders are `link_decoder("distmult", dim = 32)`,
`link_decoder("mlp", dim = 32)`, and
`link_decoder("hadamard", dim = 32)`; pass one through the `decoder`
argument.

### Ranking evaluation

Many link benchmarks rank each positive against a set of corrupted
tails.
[`corrupt_tails()`](https://adamrsophie.github.io/graphlearnR/reference/corrupt_tails.md)
draws them, filtering out known positives, and
[`metric_mrr()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
and
[`metric_hits_at_k()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
score the ranks.

``` r

neg <- corrupt_tails(ls$test_pos, num_nodes = 100, k = 20, known = ls$all_pos, seed = 1)
pos_s <- predict(fit_l, pairs = ls$test_pos, type = "score")
neg_s <- matrix(predict(fit_l, pairs = rbind(rep(ls$test_pos[1, ], each = 20), as.vector(t(neg))),
                        type = "score"), ncol = 20, byrow = TRUE)
c(mrr = metric_mrr(pos_s, neg_s), hits10 = metric_hits_at_k(pos_s, neg_s, 10))
#>       mrr    hits10 
#> 0.2736963 0.8867925
```

## Edge classification

Edge labels live in `edge_y`.
[`split_edge_labels()`](https://adamrsophie.github.io/graphlearnR/reference/split_edge_labels.md)
adds edge masks, keeping both directions of an undirected pair in the
same split. The model embeds nodes and an edge decoder classifies each
pair.

``` r

ge <- dataset_sbm(c(30, 30), p_in = 0.3, p_out = 0.05, seed = 5)
same <- ge$y[ge$edge_index[1, ]] == ge$y[ge$edge_index[2, ]]
ge$edge_y <- factor(ifelse(same, "within", "between"))
ge <- split_edge_labels(ge, seed = 5)
torch_manual_seed(5)
fit_e <- train_graph_model(gnn_model("sage", 8, 32, dropout = 0), ge,
                           "edge_classification", epochs = 60, seed = 5)
evaluate_graph_model(fit_e, ge)
#>              metric        value
#> 1          accuracy 1.000000e+00
#> 2 balanced_accuracy 1.000000e+00
#> 3   precision_macro 1.000000e+00
#> 4      recall_macro 1.000000e+00
#> 5          f1_macro 1.000000e+00
#> 6               ece 4.698913e-04
#> 7             brier 6.453759e-06
#> 8           roc_auc 1.000000e+00
```

## Training options

- `optimizer`: `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`, or a function
  of the parameters returning a torch optimizer.
- `scheduler`: `list(type = "step", step_size, gamma)`,
  `list(type = "plateau", patience, factor)`, `list(type = "cosine")`,
  or a function of the optimizer.
- `early_stopping`: patience in epochs; the best weights are restored.
- `checkpoint_path`: save the best state to disk during training.
- `loss`: `"cross_entropy"`, `"mse"`, `"l1"`, `"huber"`, `"bce"`,
  `"focal"`, or a function `(output, target)`.
- `class_weights`, `grad_clip`, `weight_decay`, `device`, and `seed`.

``` r

torch_manual_seed(6)
fit_o <- train_graph_model(gnn_model("sage", 8, 32, 3), g, "node_classification",
                           epochs = 200, optimizer = "adamw", lr = 0.02,
                           scheduler = list(type = "plateau", patience = 5, factor = 0.5),
                           early_stopping = 15, grad_clip = 1, seed = 1)
fit_o
#> <graph_fit> node_classification, 30 epochs on cpu
#>   final train loss 0.2528, val loss 0.4657, val accuracy 0.8333
#>   best epoch 15
range(fit_o$history$lr)
#> [1] 0.005 0.020
```

## Saving a model

[`save_graph_fit()`](https://adamrsophie.github.io/graphlearnR/reference/save_graph_fit.md)
writes the weights and everything needed to rebuild the model to one
`.rds` file.

``` r

path <- tempfile(fileext = ".rds")
save_graph_fit(fits$sage, path)
again <- load_graph_fit(path)
all.equal(predict(again, g, type = "prob"), predict(fits$sage, g, type = "prob"),
          tolerance = 1e-6)
#> [1] TRUE
```

Only load model files from sources you trust: an `.rds` file can hold
any R object.
