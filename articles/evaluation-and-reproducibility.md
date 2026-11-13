# Evaluation, leakage, and reproducibility

``` r

library(graphlearnR)
```

A graph learning result is only as good as its evaluation. Graphs make
it easy to leak test information into training, because examples are
connected. This article collects the rules graphlearnR follows and the
tools it gives you to report results that hold up.

## Splits that do not leak

| task | splitter | what it guarantees |
|----|----|----|
| node | [`split_nodes()`](https://adamrsophie.github.io/graphlearnR/reference/split_nodes.md) | disjoint masks; stratified by class; unlabeled nodes excluded |
| edge label | [`split_edge_labels()`](https://adamrsophie.github.io/graphlearnR/reference/split_edge_labels.md) | both directions of an undirected pair in the same split |
| link | [`split_edges()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges.md) | held out pairs removed from the message passing graph; negatives are non edges of the full graph |
| heterogeneous link | [`split_edges_hetero()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges_hetero.md) | held out edges also removed from the reverse relation |
| graph | [`split_graphs()`](https://adamrsophie.github.io/graphlearnR/reference/split_graphs.md) | random, stratified, group (no group in two splits), or scaffold |
| temporal | [`split_temporal()`](https://adamrsophie.github.io/graphlearnR/reference/split_temporal.md) | validation and test events are later than training events |

### Transductive node splits

In node classification on one graph, test nodes are present during
training as unlabeled neighbors. That is the standard transductive
setting and is not leakage, but two things would be:

- computing feature normalization statistics on all nodes. Pass
  `stats_mask = g$train_mask` to
  [`normalize_features()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md);
- selecting a model or an epoch on test accuracy. Use the validation
  mask;
  [`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
  never looks at the test mask.

``` r

g <- split_nodes(dataset_sbm(c(40, 40, 40), seed = 1), seed = 1)
g <- normalize_features(g, stats_mask = g$train_mask)
round(colMeans(g$x[g$train_mask, ]), 10)[1:4]
#> [1] 0 0 0 0
```

### Group and scaffold splits

When graphs come in related families (molecules sharing a scaffold,
samples from the same patient), a random split puts near duplicates on
both sides. Group splits keep each family in one split; scaffold splits
put the largest families in training and the rare ones in the test set,
which tests generalization to new structures.

``` r

families <- rep(c("A", "B", "C", "D", "E", "F"), times = c(30, 20, 15, 10, 15, 10))
sp <- split_graphs(length(families), method = "group", groups = families, seed = 1)
lapply(sp, function(i) unique(families[i]))
#> $train
#> [1] "A" "D" "C" "F" "E"
#> 
#> $val
#> [1] "B"
#> 
#> $test
#> character(0)
sc <- split_graphs(length(families), method = "scaffold", groups = families)
lapply(sc, function(i) unique(families[i]))
#> $train
#> [1] "A" "B" "C" "E"
#> 
#> $val
#> [1] "D"
#> 
#> $test
#> [1] "F"
```

## Metrics

[`evaluate_predictions()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_predictions.md)
returns a tidy table for a task; the individual metrics are exported
too.

``` r

truth <- factor(c("a", "a", "b", "b", "b", "c"))
prob <- rbind(c(0.7, 0.2, 0.1), c(0.4, 0.5, 0.1), c(0.1, 0.8, 0.1),
              c(0.2, 0.6, 0.2), c(0.3, 0.3, 0.4), c(0.1, 0.1, 0.8))
colnames(prob) <- levels(truth)
evaluate_predictions(truth, prob, "classification")
#>              metric     value
#> 1          accuracy 0.6666667
#> 2 balanced_accuracy 0.7222222
#> 3   precision_macro 0.7222222
#> 4      recall_macro 0.7222222
#> 5          f1_macro 0.6666667
#> 6               ece 0.3333333
#> 7             brier 0.3100000
#> 8           roc_auc 0.9629630
```

- Classification: accuracy, balanced accuracy, macro precision, recall,
  and F1, one versus rest ROC AUC, Brier score, and expected calibration
  error.
- Regression: MSE, RMSE, MAE, and R^2.
- Links: ROC AUC, average precision, MRR, and <Hits@k>; with a negative
  matrix per positive, MRR and <Hits@k> rank each positive against its
  own negatives, and ties count half.
- Clustering: normalized mutual information and adjusted Rand index.

``` r

c(nmi = metric_nmi(c(1, 1, 2, 2), c("x", "x", "y", "y")),
  ari = metric_ari(c(1, 1, 2, 2), c("x", "y", "x", "y")))
#>  nmi  ari 
#>  1.0 -0.5
```

## Report variability

A single split and a single seed can mislead, especially on small
graphs. Repeat over seeds and report a mean with a spread.

``` r

library(torch)
acc <- vapply(1:3, function(s) {
  gs <- split_nodes(dataset_sbm(c(40, 40, 40), feature_signal = 0.5, seed = 1), seed = s)
  torch_manual_seed(s)
  fit <- train_graph_model(gnn_model("sage", 8, 16, 3), gs, "node_classification",
                           epochs = 60, seed = s)
  ev <- evaluate_graph_model(fit, gs)
  ev$value[ev$metric == "accuracy"]
}, numeric(1))
c(mean = mean(acc), sd = sd(acc))
#>      mean        sd 
#> 0.9583333 0.0000000
```

## Reproducibility

Every function that draws random numbers takes a `seed` and leaves the
caller’s R random number stream untouched.

``` r

set.seed(123)
before <- .Random.seed
a <- sample_neighbors(dataset_karate(), 1:3, c(3, 3), seed = 9)
identical(.Random.seed, before)
#> [1] TRUE
identical(a$n_id, sample_neighbors(dataset_karate(), 1:3, c(3, 3), seed = 9)$n_id)
#> [1] TRUE
```

[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
seeds both R and torch. With the same seed, data, and model
initialization, two CPU runs give the same losses. GPU kernels that use
atomic additions (scatter sums) can differ in the last bits between
runs.

``` r

run <- function() {
  torch_manual_seed(7)
  m <- gnn_model("gcn", 8, 16, 3)
  train_graph_model(m, g, "node_classification", epochs = 5, seed = 7)
}
f1 <- run()
f2 <- run()
identical(f1$history$train_loss, f2$history$train_loss)
#> [1] TRUE
```

Each fit stores what is needed to report or repeat it:

``` r

str(f1$reproducibility, max.level = 1)
#> List of 16
#>  $ seed           : num 7
#>  $ task           : chr "node_classification"
#>  $ hyperparameters:List of 6
#>  $ optimizer      : chr "adam"
#>  $ scheduler      : NULL
#>  $ loss           : chr "default"
#>  $ model_config   :List of 15
#>  $ model_class    : chr "graph_model"
#>  $ num_parameters : num 467
#>  $ dataset        :List of 7
#>  $ splits         : Named int [1:3] 72 24 24
#>   ..- attr(*, "names")= chr [1:3] "train" "val" "test"
#>  $ device         : chr "cpu"
#>  $ precision      : chr "float32"
#>  $ versions       : Named chr [1:3] "4.6.2" "0.1.0" "0.17.0"
#>   ..- attr(*, "names")= chr [1:3] "R" "graphlearnR" "torch"
#>  $ platform       : chr "x86_64-pc-linux-gnu"
#>  $ time           : chr "2026-11-13 07:40:25 UTC"
```

## A checklist for papers and reports

- State the split procedure, the split sizes, and whether the setting is
  transductive or inductive.
- For link prediction, state how negatives were sampled and whether
  ranking metrics are filtered.
- For temporal data, state the cut times and confirm no future events
  are used.
- Report the mean and spread over at least three seeds, and a simple
  baseline (an MLP on features, a history baseline for dynamic links,
  the naive forecast for signals).
- Record package versions and the device, as `fit$reproducibility` does.
