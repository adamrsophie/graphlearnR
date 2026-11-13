# Continuous depth graph diffusion (graph neural ODE)

`graph_ode_block()` evolves node features by the graph diffusion
equation of GRAND (Chamberlain et al. 2021), \$\$\frac{dX}{dt} = (A - I)
X,\quad X(0) = X_0,\$\$ from \\t = 0\\ to `t_end`, and returns
\\X(t\_{end})\\.

## Usage

``` r
graph_ode_block(
  dim,
  diffusion = c("gcn", "attention"),
  t_end = 1,
  steps = 4L,
  method = c("rk4", "euler")
)

graph_ode_model(
  input_dim,
  hidden_dim = 32L,
  output_dim = 2L,
  diffusion = c("gcn", "attention"),
  t_end = 1,
  steps = 4L,
  method = c("rk4", "euler"),
  dropout = 0
)
```

## Arguments

- dim:

  Feature size (unchanged by the block).

- diffusion:

  `"gcn"` or `"attention"`.

- t_end:

  Integration time.

- steps:

  Number of integration steps.

- method:

  `"rk4"` or `"euler"`.

- input_dim, hidden_dim, output_dim:

  Feature sizes.

- dropout:

  Dropout on the encoder output.

## Value

An `nn_module`.

## Details

- `diffusion = "gcn"`: \\A = \hat D^{-1/2}(A + I)\hat D^{-1/2}\\, the
  GCN propagation matrix (linear, no parameters in the dynamics).

- `diffusion = "attention"`: \\A\_{ij} = \mathrm{softmax}\_j((W_K
  x_j)^\top W_Q x_i / \sqrt{d})\\ over the incoming edges of \\i\\ and a
  self loop, computed once from \\X_0\\ (GRAND-l).

The equation is integrated with `steps` fixed steps of explicit Euler or
classical fourth order Runge-Kutta inside the torch graph, so gradients
flow by ordinary backpropagation through the steps. This is a fixed
discretization that defines the layer, not a general ODE solver: there
is no step size control, adjoint method or error estimate. For adaptive
simulation of graph dynamics outside training, use
[`graph_diffusion()`](https://adamrsophie.github.io/graphlearnR/reference/graph_diffusion.md),
which calls modelcompR.

`graph_ode_model()` wraps the block between a linear encoder and decoder
for node level tasks and can be passed to
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md).

## References

Chamberlain, Rowbottom, Gorinova, Webb, Rossi and Bronstein (2021)
GRAND: graph neural diffusion. Chen, Rubanova, Bettencourt and Duvenaud
(2018) Neural ordinary differential equations.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  m <- graph_ode_model(34, 16, 2, t_end = 3, steps = 6)
  fit <- train_graph_model(m, g, "node_classification", epochs = 50, seed = 1)
  evaluate_graph_model(fit, g)
}
#>              metric     value
#> 1          accuracy 0.8333333
#> 2 balanced_accuracy 0.8333333
#> 3   precision_macro 0.8750000
#> 4      recall_macro 0.8333333
#> 5          f1_macro 0.8285714
#> 6               ece 0.1445739
#> 7             brier 0.2138200
#> 8           roc_auc 1.0000000
#> 9 average_precision 1.0000000
```
