# Attention weights of attention layers

Collects the attention coefficients stored by attention layers
(`gat_conv`, `gatv2_conv`, `transformer_conv`) during the last forward
pass. Run a forward pass (for example
[`predict()`](https://rdrr.io/r/stats/predict.html)) first. Attention
shows how a layer weighted neighbors; it is not a feature importance
measure.

## Usage

``` r
attention_weights(model)
```

## Arguments

- model:

  A model or `graph_fit`.

## Value

A list, one data frame per attention layer, with `from`, `to` and one
column per head.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gat", 34, 8, 2, heads = 2), g,
                           "node_classification", epochs = 5, seed = 1)
  invisible(predict(fit, g))
  head(attention_weights(fit)[[1]])
}
#>   from to      head1      head2
#> 1    1  2 0.12065870 0.09890987
#> 2    2  1 0.04900805 0.05869930
#> 3    1  3 0.12101147 0.09048011
#> 4    3  1 0.04941681 0.05875746
#> 5    1  4 0.21639322 0.14588362
#> 6    4  1 0.05306524 0.05890427
```
