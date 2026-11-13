# Link and edge decoders

`link_decoder()` scores node pairs from node embeddings `z`:

## Usage

``` r
link_decoder(
  type = c("dot", "distmult", "mlp", "hadamard"),
  dim = NULL,
  num_relations = 1L,
  hidden = 64L
)

edge_decoder(dim, out_dim, edge_dim = 0L, hidden = 64L)
```

## Arguments

- type:

  Decoder type.

- dim:

  Embedding size (not needed for `"dot"`).

- num_relations:

  Relations for `"distmult"`.

- hidden:

  Hidden size of MLP decoders.

- out_dim:

  Output size of `edge_decoder()`.

- edge_dim:

  Edge feature size appended to the pair (0 for none).

## Value

An `nn_module` called as `decoder(z, pairs, relation = NULL)`, where
`pairs` is a 2 by K long tensor of one based node indices. For typed
links pass `z = list(z_src, z_dst)`.

## Details

- `"dot"`: \\z_u^\top z_v\\;

- `"distmult"`: \\z_u^\top \mathrm{diag}(r) z_v\\ with one learned
  vector per relation (DistMult, Yang et al. 2015);

- `"mlp"`: an MLP on \\\[z_u \Vert z_v\]\\ (order sensitive, so suited
  to directed links);

- `"hadamard"`: an MLP on \\z_u \odot z_v\\.

Scores are logits; apply a sigmoid for probabilities. `edge_decoder()`
is an MLP on \\\[z_u \Vert z_v \Vert e\_{uv}\]\\ for edge classification
or regression.

## Examples

``` r
if (torch::torch_is_installed()) {
  dec <- link_decoder("distmult", dim = 8)
  z <- torch::torch_randn(5, 8)
  pairs <- torch::torch_tensor(rbind(c(1, 2), c(3, 4)), dtype = torch::torch_long())
  dec(z, pairs)
}
#> torch_tensor
#> -0.2736
#>  3.1483
#> [ CPUFloatType{2} ][ grad_fn = <SumBackward1> ]
```
