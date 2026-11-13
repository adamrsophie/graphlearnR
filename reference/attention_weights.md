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
