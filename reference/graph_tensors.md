# Convert graph data to torch tensors

Converts the stored R values once, so a training loop can reuse the same
tensors on one device. Node indices stay one based, which is what R
`torch` index operations expect. Factor and integer labels become `long`
tensors (one based class codes, as
[`torch::nnf_cross_entropy()`](https://torch.mlverse.org/docs/reference/nnf_cross_entropy.html)
expects); numeric labels become floating point tensors.

## Usage

``` r
graph_tensors(g, device = NULL, dtype = torch_float())
```

## Arguments

- g:

  A `graph_data` or `graph_batch` object.

- device:

  A `torch_device`, or one of `"cpu"`, `"cuda"`, `"mps"`, `"auto"`.

- dtype:

  Floating point dtype for features.

## Value

A list of tensors with class `graph_tensors`: `x`, `edge_index` (2 by E,
long), `edge_weight`, `edge_attr`, `y`, `edge_y`, `graph_y`, `batch`
(graph membership for batches), masks, and `num_nodes`.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- graph_data(x = diag(3), edge_index = rbind(1:2, 2:3))
  tt <- graph_tensors(g)
  tt$edge_index
}
#> torch_tensor
#>  1  2
#>  2  3
#> [ CPULongType{2,2} ]
```
