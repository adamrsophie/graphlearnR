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
#> 9 average_precision 1.0000000
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
#> [1] 0.006706327
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
#> 32   32   32  1  -0.6392072
#> 22   22   13  1   0.6088946
#> 20   20   12  1   0.5910738
#> 12   12    7  1   0.4951162
#> 10   10    6  1   0.4866576
#> 8     8    5  1   0.4837199
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
#> 11   22   13  1   0.6273994
#> 10   20   12  1   0.6089315
#> 6    12    7  1   0.5063629
#> 5    10    6  1   0.4981394
#> 4     8    5  1   0.4943409
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

### PGExplainer

[`pg_explainer()`](https://adamrsophie.github.io/graphlearnR/reference/pg_explainer.md)
(Luo et al. 2020) trains one small network that predicts the edge mask
of any target from the model’s node embeddings, so after training each
new explanation costs one forward pass. It is trained here on the
training nodes and then applied to node 1.

``` r

pg <- pg_explainer(fit, g, epochs = 20, edge_size = 0.05, seed = 1)
px <- predict(pg, g, target = 1)
head(px$edges[order(-px$edges$attribution), ])
#>    edge from to  attribution
#> 83   83    9 31 4.340116e-07
#> 84   84   31  9 4.168331e-07
#> 59   59    3 10 3.237934e-07
#> 60   60   10  3 2.779633e-07
#> 55   55    3 29 2.117927e-07
#> 61   61    3  9 2.101551e-07
range(px$edges$attribution)
#> [1] 4.108396e-11 4.340116e-07
```

The mask values are all close to zero. That is the answer to the
question PGExplainer asks (the smallest set of edges that keeps the
prediction), not a failure: with one-hot node features this model has
memorized the nodes, and it predicts the same class for every node even
with all edges removed.

``` r

tt <- graph_tensors(g)
m <- fit$model
m$eval()
full <- with_no_grad(m(tt$x, tt$edge_index))
no_edges <- with_no_grad(m(tt$x, tt$edge_index, edge_weight = torch_zeros(num_edges(g))))
mean(as.integer(full$argmax(dim = 2)) == as.integer(no_edges$argmax(dim = 2)))
#> [1] 1
```

Only the order of the mask values carries information here, which is
what
[`explanation_fidelity()`](https://adamrsophie.github.io/graphlearnR/reference/explanation_fidelity.md)
below uses. On a model whose predictions need the graph, the mask
separates the edges that matter; the tests of the package check that it
recovers the one path that carries the signal in a toy graph.

### SubgraphX

[`subgraphx()`](https://adamrsophie.github.io/graphlearnR/reference/subgraphx.md)
(Yuan et al. 2021) searches, by Monte Carlo tree search, for a small
connected subgraph around the target whose Shapley value (its average
contribution when added to random coalitions of nearby nodes) is
largest. It removes nodes instead of reweighting edges, so it also
explains models that ignore `edge_weight`, at the price of many forward
passes.

``` r

sx <- subgraphx(fit, g, target = 1, max_nodes = 5, rollouts = 5, samples = 8, seed = 1)
sx$subgraph_nodes
#> [1] 1 2 3 4 8
sx$score
#> [1] 1.045696
head(sx$candidates)
#>    size       nodes     score
#> 3     5   1,2,3,4,8 1.0456963
#> 5     5  1,2,4,8,14 0.9866302
#> 4     5 1,2,4,14,34 0.9631625
#> 8     5  1,3,4,8,14 0.8916118
#> 10    5 1,3,8,14,33 0.5923036
#> 2     5  1,2,3,4,14 0.4377630
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
#> 3    1  3 0.16340749 0.13701235
#> 4    3  1 0.05869362 0.03081638
#> 5    1  4 0.22324155 0.21166605
#> 6    4  1 0.12893747 0.06319855
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
      gnn_explainer = explanation_fidelity(fit, g, gx, k = 5),
      pg_explainer = explanation_fidelity(fit, g, px, k = 5),
      subgraphx = explanation_fidelity(fit, g, sx, k = sum(sx$edges$attribution)))
#>               fidelity_plus fidelity_minus
#> gradient         1.72813797      -3.426292
#> gnn_explainer   -0.30329800       3.899850
#> pg_explainer    -0.02089596       3.899850
#> subgraphx        0.44322872       3.091657
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
