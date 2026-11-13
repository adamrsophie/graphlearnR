# Personalized PageRank propagation (APPNP)

`appnp()` propagates features with the approximate personalized PageRank
iteration of Klicpera, Bojchevski and Günnemann (2019), \$\$H^{(0)} =
X,\quad H^{(k+1)} = (1 - \alpha) \hat A H^{(k)} + \alpha X,\$\$ for `K`
steps, with \\\hat A\\ the GCN propagation matrix (self loops added,
symmetric normalization). It has no parameters. As `K` grows the result
approaches \\\alpha (I - (1 - \alpha)\hat A)^{-1} X\\.

## Usage

``` r
appnp(K = 10L, alpha = 0.1, dropout = 0)

appnp_model(
  input_dim,
  hidden_dim = 64L,
  output_dim = 2L,
  K = 10L,
  alpha = 0.1,
  dropout = 0.5
)
```

## Arguments

- K:

  Propagation steps.

- alpha:

  Teleport (restart) probability.

- dropout:

  For `appnp()`, dropout on the propagation weights during training; for
  `appnp_model()`, dropout in the MLP.

- input_dim, hidden_dim, output_dim:

  Feature sizes of the MLP.

## Value

An `nn_module`.

## Details

`appnp_model()` is the full APPNP model: an MLP predicts per node and
the predictions are then propagated. It can be passed to
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
for node tasks.

## References

Klicpera, Bojchevski and Günnemann (2019) Predict then propagate: graph
neural networks meet personalized PageRank.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  m <- appnp_model(34, 16, 2, K = 10, alpha = 0.1)
  fit <- train_graph_model(m, g, "node_classification", epochs = 50, seed = 1)
  evaluate_graph_model(fit, g)
}
#>              metric     value
#> 1          accuracy 0.8333333
#> 2 balanced_accuracy 0.8333333
#> 3   precision_macro 0.8750000
#> 4      recall_macro 0.8333333
#> 5          f1_macro 0.8285714
#> 6               ece 0.1373674
#> 7             brier 0.2635723
#> 8           roc_auc 1.0000000
```
