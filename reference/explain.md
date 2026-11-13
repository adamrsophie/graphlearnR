# Explain node predictions

Post hoc explanations of a trained model's output for one target node
(or one graph, with `target = NULL` and a graph level model). They
describe **how the model's output responds to its inputs**; they are not
causal explanations of the data generating process, and attention
weights alone are not importance scores.

## Usage

``` r
explain_saliency(model, g, target = NULL, output = NULL, times_input = FALSE)

explain_integrated_gradients(
  model,
  g,
  target = NULL,
  output = NULL,
  baseline = 0,
  steps = 50L
)

explain_edges(model, g, target = NULL, output = NULL)

explain_occlusion(model, g, target = NULL, output = NULL)

gnn_explainer(
  model,
  g,
  target = NULL,
  output = NULL,
  epochs = 100L,
  lr = 0.01,
  edge_size = 0.005,
  feature_size = 1,
  entropy = 1,
  seed = NULL
)
```

## Arguments

- model:

  A trained model (or a `graph_fit`).

- g:

  A `graph_data` object.

- target:

  Target node index, or `NULL` for a graph level output.

- output:

  Output column to explain; default the predicted class.

- times_input:

  Multiply gradients by the input.

- baseline:

  Baseline features: a number or a matrix like `g$x`.

- steps:

  Riemann sum points.

- epochs:

  Optimization steps.

- lr:

  Learning rate.

- edge_size, feature_size:

  Sparsity penalties.

- entropy:

  Mask entropy penalty, pushing mask values towards 0 or 1.

- seed:

  Optional integer seed.

## Value

A list with attribution matrices or vectors and metadata, of class
`graph_explanation`.

## Details

- `explain_saliency()`: gradient of the selected output with respect to
  node features (Simonyan et al. 2014), optionally times the input.

- `explain_integrated_gradients()`: integrated gradients (Sundararajan
  et al. 2017) along a straight path from `baseline` to the features
  with a Riemann sum of `steps` points. The result includes
  `completeness_gap`, the difference between the summed attributions and
  `f(x) - f(baseline)`; a large gap means more steps are needed.

- `explain_edges()`: gradient of the output with respect to a
  multiplicative mask on edge weights, evaluated at mask 1. Needs a
  model whose layers use `edge_weight` (`"gcn"`, `"sage"`,
  `"graph_conv"`, `"gin"`).

- `explain_occlusion()`: change in the output when each feature column,
  or each incoming edge of the target, is removed.

- `gnn_explainer()`: learns a soft edge mask and feature mask that keep
  the prediction while staying sparse (Ying et al. 2019).

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
                           "node_classification", epochs = 20, seed = 1)
  ex <- explain_edges(fit, g, target = 1)
  head(ex$edges[order(-abs(ex$edges$attribution)), ])
}
#>     edge from to attribution
#> 32    32   32  1 -0.05562937
#> 20    20   12  1  0.04393630
#> 16    16    9  1 -0.04008571
#> 22    22   13  1  0.03602337
#> 106  106   34 20 -0.02254511
#> 26    26   18  1  0.02136113
```
