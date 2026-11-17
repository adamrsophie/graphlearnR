# Linear (kernelized) attention within each graph

Multi head self attention among the nodes of each graph of a batch whose
cost is linear in the number of nodes. The softmax kernel \\\exp(q^\top
k / \sqrt{C})\\ is replaced by a feature map product \\\phi(q)^\top
\phi(k)\\, so for node \\i\\ of graph \\g\\ \$\$o_i =
\frac{\phi(q_i)^\top \sum\_{j \in g} \phi(k_j) v_j^\top} {\phi(q_i)^\top
\sum\_{j \in g} \phi(k_j)}.\$\$ With `kernel = "favor"` (Performer,
Choromanski et al. 2021), \\\phi(x) = m^{-1/2} \exp(W^\top x' - \lVert
x' \rVert^2 / 2)\\ with \\x' = x / C^{1/4}\\ and `num_features` fixed
Gaussian random features \\W\\, an unbiased estimate of the softmax
kernel; with `kernel = "elu"` (Katharopoulos et al. 2020), \\\phi(x) =
\mathrm{elu}(x) + 1\\. The per graph sums use scatter operations, so
graphs are never padded and never attend to each other. Memory is \\O(N
H m C)\\.

## Usage

``` r
linear_attention(
  dim,
  heads = 4L,
  num_features = 64L,
  kernel = c("favor", "elu")
)
```

## Arguments

- dim:

  Feature size.

- heads:

  Attention heads (`dim` must be divisible by `heads`).

- num_features:

  Random features \\m\\ for `"favor"`.

- kernel:

  `"favor"` or `"elu"`.

## Value

An `nn_module` called as `attn(x, batch = NULL)`, returning N by `dim`.

## References

Choromanski et al. (2021) Rethinking attention with Performers. ICLR.
Katharopoulos et al. (2020) Transformers are RNNs. ICML.

## Examples

``` r
if (torch::torch_is_installed()) {
  attn <- linear_attention(8, heads = 2)
  attn(torch::torch_randn(5, 8))$shape
}
#> [1] 5 8
```
