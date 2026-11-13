# Heterogeneous graph transformer convolution

One layer of the heterogeneous graph transformer (HGT, Hu et al. 2020).
For a target node \\t\\ of type \\\tau(t)\\, an incoming edge from \\s\\
through relation \\r\\, and head \\h\\ with \\D\\ channels, \$\$k_s =
K\_{\tau(s)} x_s,\\ q_t = Q\_{\tau(t)} x_t,\\ v_s = V\_{\tau(s)}
x_s,\$\$ \$\$e\_{st}^h = (k_s^h W^{att}\_{r,h}) \cdot q_t^h \\
\mu\_{r,h} / \sqrt{D},\$\$ \$\$\alpha\_{st}^h = \mathrm{softmax}\_{s \in
N(t)}(e\_{st}^h),\$\$ where the softmax runs over **all** incoming edges
of \\t\\, across relations. The update is \$\$\tilde h_t = \sum_s
\alpha\_{st} (v_s W^{msg}\_{r}),\quad h_t' = \beta\_{\tau(t)}
A\_{\tau(t)}(\mathrm{GELU}(\tilde h_t)) + (1 - \beta\_{\tau(t)})
x_t,\$\$ with \\\beta = \mathrm{sigmoid}(\text{skip})\\ a learned gate
per type (the skip term is used only when input and output sizes match).
Node types without incoming edges get \\A\_{\tau}(0)\\ plus the skip
term. Relative temporal encoding from the paper is not implemented.

## Usage

``` r
hgt_conv(node_types, relations, in_features, out_features, heads = 2L)
```

## Arguments

- node_types:

  Character vector of node types.

- relations:

  Character vector of relation keys `"src__rel__dst"`.

- in_features:

  Input size, the same for every node type.

- out_features:

  Output size; must be divisible by `heads`.

- heads:

  Number of attention heads.

## Value

An `nn_module` called as `layer(x_dict, edge_index_dict)` that returns a
named list of node feature tensors.

## Details

Cost is \\O(E H D^2)\\ for the relation transforms plus \\O(N F F')\\
for the type projections; the layer keeps an E by heads attention tensor
per target type in `last_attention`.

## References

Hu, Dong, Wang and Sun (2020) Heterogeneous graph transformer.
Proceedings of The Web Conference.

## Examples

``` r
if (torch::torch_is_installed()) {
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
  ht <- hetero_tensors(hg)
  layer <- hgt_conv(hg$node_types, names(hg$edges), 8, 8, heads = 2)
  x <- list(drug = torch::torch_randn(60, 8), protein = torch::torch_randn(80, 8))
  lapply(layer(x, ht$edge_index), dim)
}
#> $drug
#> [1] 60  8
#> 
#> $protein
#> [1] 80  8
#> 
```
