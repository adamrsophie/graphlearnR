# Heterogeneous convolution wrapper

Applies one bipartite layer per relation and sums (or averages) the
results arriving at each target node type, the scheme of PyG's
`HeteroConv`. Node types that receive no relation keep their input
features.

## Usage

``` r
hetero_conv(
  convs,
  aggr = c("sum", "mean", "semantic"),
  dim = NULL,
  semantic_dim = 128L
)
```

## Arguments

- convs:

  Named list of layers keyed by relation key; each is called as
  `layer(list(x_src, x_dst), edge_index)`.

- aggr:

  `"sum"`, `"mean"` or `"semantic"` over relations.

- dim:

  Output size of the layers, needed for `"semantic"`.

- semantic_dim:

  Hidden size of the semantic attention.

## Value

An `nn_module` called as `module(x_dict, edge_index_dict)`.

## Details

With `aggr = "semantic"`, relations arriving at a node type are combined
by the semantic attention of HAN (Wang et al. 2019): relation \\r\\ gets
the score \\w_r = \frac{1}{N}\sum_i q^\top \tanh(W z_i^{r} + b)\\ (one
\\q, W, b\\ per target type), \\\beta = \mathrm{softmax}(w)\\, and the
output is \\\sum_r \beta_r z^{r}\\. The last weights are kept in
`module$last_semantic_attention` for inspection (they show how the layer
weighted relations, not causal importance).

## References

Wang et al. (2019) Heterogeneous graph attention network. The Web
Conference (WWW).

## Examples

``` r
if (torch::torch_is_installed()) {
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
  convs <- lapply(stats::setNames(nm = edge_types(hg)), function(r) sage_conv(16, 8))
  conv <- hetero_conv(convs)
  conv
}
#> An `nn_module` containing 1,584 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • convs: <nn_module> #1,584 parameters
```
