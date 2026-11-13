# Principal neighborhood aggregation (PNA)

The PNA layer of Corso, Cavalleri, Beaini, Liò and Veličković (2020), in
the single tower form of PyTorch Geometric's `PNAConv` (one pre and one
post layer, `divide_input = FALSE`). For node \\i\\ with in degree
\\d_i\\:

## Usage

``` r
pna_conv(
  in_features,
  out_features,
  deg,
  aggregators = c("mean", "min", "max", "std"),
  scalers = c("identity", "amplification", "attenuation"),
  edge_dim = NULL
)

pna_degree(graphs)
```

## Arguments

- in_features, out_features:

  Feature sizes.

- deg:

  Degree histogram from `pna_degree()`.

- aggregators:

  Any of `"mean"`, `"min"`, `"max"`, `"std"`, `"sum"`.

- scalers:

  Any of `"identity"`, `"amplification"`, `"attenuation"`, `"linear"`,
  `"inverse_linear"`.

- edge_dim:

  Edge feature size, or `NULL`.

- graphs:

  A `graph_data` object or a list of them.

## Value

`pna_conv()`: an `nn_module` called as
`layer(x, edge_index, edge_attr = NULL)`. `pna_degree()`: an integer
vector.

## Details

1.  messages \\m\_{ji} = U \[x_i \Vert x_j \Vert E e\_{ji}\]\\ (the edge
    term only with `edge_dim`);

2.  each aggregator (mean, min, max, std, sum) is applied to the
    incoming messages and each result is multiplied by each scaler:
    identity, amplification \\\log(d_i + 1) / \delta\\, attenuation
    \\\delta / \log(d_i + 1)\\, linear \\d_i / \bar d\\ or inverse
    linear \\\bar d / d_i\\, with \\d_i\\ clamped to at least 1 and
    \\\delta\\, \\\bar d\\ the mean of \\\log(d + 1)\\ and of \\d\\ over
    the training degree histogram `deg`;

3.  \\x_i' = W_2 \\ M \[x_i \Vert \text{scaled aggregates}\]\\.

The std aggregator is \\\sqrt{\max(\mathrm{var}, 0) + 10^{-5}}\\, so
nodes without incoming edges, or with identical incoming messages, get
\\\sqrt{10^{-5}}\\ from it and zeros from every other aggregator. Other
implementations may treat zero variance differently.

`pna_degree()` computes the in degree histogram `deg` from training
graphs: element \\d + 1\\ counts nodes with in degree \\d\\.

Input `x` is N by F; output N by F'. No self loops are added; edge input
is `edge_attr` when `edge_dim` is set. Cost: \\O(E F^2)\\ for the
message transform, \\O(a E F)\\ for `a` aggregators, and \\O(N a s F
F')\\ for the post transform with `s` scalers; memory holds E by F
messages and the N by \\(a s + 1) F\\ concatenation.

The benchmark workflow (`benchmarks/pyg/`) copies the weights of a layer
with the mean, min, max and sum aggregators and all five scalers, and of
one with the std aggregator, into PyTorch Geometric's `PNAConv` (one
tower, one pre and post layer) and checks that the outputs agree (the
first agreed to 2e-6 on CI).

## References

Corso, Cavalleri, Beaini, Liò and Veličković (2020) Principal
neighbourhood aggregation for graph nets.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- dataset_karate()
  tt <- graph_tensors(g)
  layer <- pna_conv(34, 8, deg = pna_degree(g))
  layer(tt$x, tt$edge_index)$shape
}
#> [1] 34  8
```
