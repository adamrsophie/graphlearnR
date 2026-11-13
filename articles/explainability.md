# Explaining graph neural network predictions

``` r

library(graphlearnR)
library(torch)
```

An explanation answers: which node features and which edges made the
model predict this? graphlearnR implements gradient based, perturbation
based, and optimization based methods, plus two checks of explanation
quality. All methods take a trained `graph_fit` (or a model), a graph,
and a target node; leave `target = NULL` for a graph level model.

## A model to explain

``` r

g <- split_nodes(dataset_karate(), seed = 1)
torch_manual_seed(1)
fit <- train_graph_model(gnn_model("gcn", 34, 16, 2, dropout = 0), g,
                         "node_classification", epochs = 100, seed = 1)
evaluate_graph_model(fit, g)
#>              metric     value
#> 1          accuracy 0.8333333
#> 2 balanced_accuracy 0.8333333
#> 3   precision_macro 0.8750000
#> 4      recall_macro 0.8333333
#> 5          f1_macro 0.8285714
#> 6               ece 0.1659179
#> 7             brier 0.3303945
#> 8           roc_auc 1.0000000
```

## Feature attributions

### Saliency

The gradient of the selected output with respect to the input features.
By default the output is the predicted class of the target node.

``` r

sal <- explain_saliency(fit, g, target = 1)
sal
#> <graph_explanation> saliency for node 1, output 1
top <- order(-abs(sal$node_feature))[1:5]
arrayInd(top, dim(sal$node_feature))
#>      [,1] [,2]
#> [1,]    1   13
#> [2,]    1   33
#> [3,]    1   14
#> [4,]    1    7
#> [5,]    1    4
```

With identity features, feature j of node i is “being node i”, so the
largest saliencies point at the nodes whose identity most affects node
1’s prediction.

### Integrated gradients

Saliency is local. Integrated gradients (Sundararajan, Taly, and Yan
2017) average gradients along the straight path from a baseline (zero by
default) to the input and multiply by the difference. The attributions
then add up to the change in output between baseline and input;
graphlearnR reports the remaining gap of the Riemann approximation.

``` r

ig <- explain_integrated_gradients(fit, g, target = 1, steps = 64)
sum(ig$node_feature)
#> [1] 9.833233
ig$completeness_gap
#> [1] 0.006706059
```

## Edge attributions

### Edge gradients

[`explain_edges()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
attaches a weight of one to every edge, takes the gradient of the output
with respect to those weights, and returns a table.

``` r

eg <- explain_edges(fit, g, target = 1)
head(eg$edges[order(-abs(eg$edges$attribution)), ])
#>    edge from to attribution
#> 32   32   32  1  -0.6392071
#> 22   22   13  1   0.6088946
#> 20   20   12  1   0.5910739
#> 12   12    7  1   0.4951163
#> 10   10    6  1   0.4866577
#> 8     8    5  1   0.4837200
```

### Occlusion

[`explain_occlusion()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
removes each incoming edge of the target in turn and records how much
the output drops. It also masks each feature column. It is slow for many
edges but needs no gradients and is easy to interpret.

``` r

oc <- explain_occlusion(fit, g, target = 1)
head(oc$edges[order(-oc$edges$attribution), ])
#>    edge from to attribution
#> 11   22   13  1   0.6274004
#> 10   20   12  1   0.6089334
#> 6    12    7  1   0.5063639
#> 5    10    6  1   0.4981394
#> 4     8    5  1   0.4943419
#> 9    18   11  1   0.4860201
```

### GNNExplainer

[`gnn_explainer()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
(Ying et al. 2019) learns a soft mask over edges and features that keeps
the prediction while staying small and close to binary. The trained
model’s weights are frozen during the optimization and restored
afterwards.

``` r

gx <- gnn_explainer(fit, g, target = 1, epochs = 100, seed = 1)
head(gx$edges[order(-gx$edges$attribution), ])
#>     edge from to attribution
#> 121  121   24 34   0.9116459
#> 45    45    2 22   0.9104612
#> 56    56   29  3   0.9084590
#> 96    96   34 15   0.9083152
#> 33    33    2  3   0.9076678
#> 124  124   30 24   0.9075592
range(gx$edges$attribution)
#> [1] 0.3414106 0.9116459
```

### Explanation subgraphs

[`explanation_subgraph()`](https://adamrsophie.github.io/graphlearnR/reference/explanation_subgraph.md)
turns any edge attribution into a small graph: the `k` most important
edges (both directions of an undirected pair count as one), the nodes
they touch, and the target. Its `node_ids` map back to the original
graph, so it can be plotted or passed to the model again.

``` r

sub <- explanation_subgraph(g, gx, k = 5)
sub
#> <graph_data> undirected graph: 8 nodes, 10 edges (5 undirected pairs)
#>   node features [8 x 34]
#>   node labels: factor with 2 levels
#>   masks: train=6, val=0, test=2
sub$node_ids
#> [1]  1  2  3 15 22 24 29 34
```

## Attention

For attention models,
[`attention_weights()`](https://adamrsophie.github.io/graphlearnR/reference/attention_weights.md)
collects the coefficients of the last forward pass of each attention
layer. Attention is a model component, not an explanation; high
attention does not have to mean high influence.

``` r

torch_manual_seed(2)
fit_gat <- train_graph_model(gnn_model("gat", 34, 8, 2, heads = 2), g,
                             "node_classification", epochs = 50, seed = 1)
invisible(predict(fit_gat, g))
aw <- attention_weights(fit_gat)
length(aw)
#> [1] 2
head(aw[[1]])
#>   from to      head1      head2
#> 1    1  2 0.19154441 0.16146892
#> 2    2  1 0.03586031 0.07413401
#> 3    1  3 0.16340747 0.13701235
#> 4    3  1 0.05869362 0.03081637
#> 5    1  4 0.22324155 0.21166605
#> 6    4  1 0.12893744 0.06319855
```

## Is the explanation any good?

### Fidelity

[`explanation_fidelity()`](https://adamrsophie.github.io/graphlearnR/reference/explanation_fidelity.md)
keeps or removes the `k` most important edges and measures the change in
the predicted probability (Yuan et al. 2022). `fidelity_plus` is the
drop when the important edges are removed (higher is better);
`fidelity_minus` is the drop when only they are kept (lower is better).

``` r

rbind(gradient = explanation_fidelity(fit, g, eg, k = 5),
      gnn_explainer = explanation_fidelity(fit, g, gx, k = 5))
#>               fidelity_plus fidelity_minus
#> gradient          1.7281389      -3.426291
#> gnn_explainer    -0.3032961       3.899852
```

### Stability

[`explanation_stability()`](https://adamrsophie.github.io/graphlearnR/reference/explanation_fidelity.md)
perturbs the features with small noise, explains again, and reports the
rank correlation with the original explanation. A low value means the
explanation would change for inputs that are practically the same.

``` r

explanation_stability(fit, g, function(m, h) explain_edges(m, h, target = 1),
                      noise = 0.01, reps = 3, seed = 1)
#> mean_spearman  min_spearman 
#>     0.9977885     0.9963462
```

## Caveats

- Gradient methods explain the model, not the data. Two models with the
  same accuracy can have very different explanations.
- With one hot identity features, feature attributions are about node
  identity; with real features, they are about feature values.
- Always check fidelity before you interpret an explanation, and compare
  against a random edge ranking.
