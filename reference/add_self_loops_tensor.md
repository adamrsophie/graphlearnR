# Self loop and normalization helpers

Tensor level helpers used by the layers. `add_self_loops_tensor()`
appends one self loop per node; with `remaining = TRUE` nodes that
already have a self loop keep theirs and get no second one. `gcn_norm()`
computes the symmetric normalization \\w\_{ij} / \sqrt{d_i d_j}\\ with
\\d\\ the weighted in degree after adding self loops, as in Kipf and
Welling (2017).

## Usage

``` r
add_self_loops_tensor(
  edge_index,
  num_nodes,
  edge_weight = NULL,
  fill_value = 1,
  remaining = FALSE,
  edge_attr = NULL
)

remove_self_loops_tensor(edge_index, edge_weight = NULL, edge_attr = NULL)

gcn_norm(
  edge_index,
  num_nodes,
  edge_weight = NULL,
  add_self_loops = TRUE,
  improved = FALSE,
  dtype = NULL
)
```

## Arguments

- edge_index:

  Long tensor, 2 by E.

- num_nodes:

  Number of nodes.

- edge_weight:

  Optional weight tensor of length E.

- fill_value:

  Weight given to the new self loops.

- remaining:

  Only add loops to nodes that do not have one.

- edge_attr:

  Optional edge feature tensor; new loops get the mean edge feature of
  their incoming edges (zero if none).

- add_self_loops:

  Add self loops before normalizing.

- improved:

  Use fill value 2 for self loops (Kipf and Welling's "improved"
  variant).

- dtype:

  Floating point type of the weights (default float32); layers pass the
  dtype of their input so double precision models normalize in double
  precision.

## Value

A list with `edge_index`, `edge_weight` and `edge_attr`.

## Examples

``` r
if (torch::torch_is_installed()) {
  ei <- torch::torch_tensor(rbind(c(1, 2), c(2, 3)), dtype = torch::torch_long())
  add_self_loops_tensor(ei, 3)$edge_index
  gcn_norm(ei, 3)$edge_weight
}
#> torch_tensor
#>  0.7071
#>  0.5000
#>  1.0000
#>  0.5000
#>  0.5000
#> [ CPUFloatType{5} ]
```
