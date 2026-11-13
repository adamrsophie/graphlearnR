# Working with modelcompR, symcompR and other sibling packages

``` r

library(graphlearnR)
```

graphlearnR is one of a series of R packages by the same author. The
graph learning engine depends only on `torch`, `igraph` and `Matrix`;
every connection to a sibling package is optional, lives in `Suggests`,
and fails with a clear message when the sibling is not installed. This
article shows the two bridges that exist today and records what was
investigated for the others and why it was deferred.

The code chunks below run only when the sibling package is installed. On
the package website they ran if the build machine had it.

## Graph dynamics with modelcompR

[modelcompR](https://github.com/adamrsophie/modelcompR) solves
differential equations.
[`graph_diffusion()`](https://adamrsophie.github.io/graphlearnR/reference/graph_diffusion.md)
uses it to simulate heat diffusion on a graph, \frac{du}{dt} = -\kappa L
u, where L is the graph Laplacian. graphlearnR builds the operator and
modelcompR integrates it; graphlearnR has no ODE solver of its own.

``` r

g <- graph_data(edge_index = rbind(c(1, 2, 3, 3), c(2, 3, 4, 5)), num_nodes = 5,
                directed = FALSE)
u <- graph_diffusion(g, u0 = c(1, 0, 0, 0, 0), times = c(0, 0.5, 1, 2, 5, 20))
round(u, 3)
#>          1     2     3     4     5
#> [1,] 1.000 0.000 0.000 0.000 0.000
#> [2,] 0.674 0.257 0.051 0.009 0.009
#> [3,] 0.522 0.303 0.100 0.037 0.037
#> [4,] 0.378 0.280 0.148 0.097 0.097
#> [5,] 0.237 0.218 0.189 0.178 0.178
#> [6,] 0.200 0.200 0.200 0.200 0.200
#> attr(,"times")
#> [1]  0.0  0.5  1.0  2.0  5.0 20.0
rowSums(u)
#> [1] 1 1 1 1 1 1
```

Heat flows from node 1 to the rest; the total stays at one because the
plain Laplacian conserves it on an undirected graph, and every node
tends to the mean, 0.2. The test suite checks the solution against the
closed form e^{-\kappa t L} u_0 computed from an eigendecomposition.

This is a mechanistic simulation tool. modelcompR solves on R numerics,
so gradients do not flow back into torch, and
[`graph_diffusion()`](https://adamrsophie.github.io/graphlearnR/reference/graph_diffusion.md)
is not a neural ODE layer. Diffusion outputs can still be used as fixed
node features, for example heat kernel signatures at a few times:

``` r

heat <- sapply(seq_len(num_nodes(g)), function(i) {
  e <- numeric(num_nodes(g))
  e[i] <- 1
  graph_diffusion(g, e, times = c(0, 1))[2, i]
})
round(heat, 3)
#>     1     2     3     4     5 
#> 0.522 0.320 0.244 0.495 0.495
```

Node 3, the hub, keeps the least of its own heat after one time unit.

## Exact operators with symcompR

[symcompR](https://github.com/adamrsophie/symcompR) is a computer
algebra system.
[`graph_operator_symbolic()`](https://adamrsophie.github.io/graphlearnR/reference/graph_operator_symbolic.md)
returns the adjacency matrix, the Laplacian, or the GCN propagation
matrix \hat D^{-1/2}(A + I)\hat D^{-1/2} with exact entries, so the
normalization a layer applies can be read off directly.

``` r

path <- graph_data(edge_index = rbind(1:3, 2:4), num_nodes = 4, directed = FALSE)
P <- graph_operator_symbolic(path, "gcn")
P
#> Matrix(4, 4)
#> [1/2      , sqrt(6)/6, 0        , 0        ]
#> [sqrt(6)/6, 1/3      , 1/3      , 0        ]
#> [0        , 1/3      , 1/3      , sqrt(6)/6]
#> [0        , 0        , sqrt(6)/6, 1/2      ]
```

The square roots stay exact.
[`as.matrix()`](https://rdrr.io/r/base/matrix.html) gives the numeric
values, which match what
[`gcn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
computes when its weight is the identity:

``` r

layer <- gcn_conv(4, 4, bias = FALSE)
invisible(torch::with_no_grad(layer$lin$weight$copy_(torch::torch_eye(4))))
out <- as.matrix(layer(torch::torch_eye(4), graph_tensors(path)$edge_index))
all.equal(out, as.matrix(P), tolerance = 1e-6, check.attributes = FALSE)
#> [1] TRUE
```

The bridge targets small graphs: exact arithmetic is slow and each entry
is a symbolic object.

## Investigated and deferred

The design brief asks to “investigate” connections with jaxR, optcompR
and z3r, and to keep all of them “optional”. What follows is the outcome
of that investigation for this release.

- **jaxR.** jaxR offers composable transformations (`grad`, `jit`,
  `vmap`) and segment reductions (`ops_segment_sum` and others) that
  could back the scatter kernels. A second backend needs a separation of
  tensor operations, model definition, differentiation and devices that
  graphlearnR does not have yet, and the brief warns against porting
  torch models mechanically to another backend. Deferred until both
  packages have a stable interface for exchanging arrays.
- **optcompR.** optcompR covers mathematical optimization, including
  mixed integer programs. Candidate uses are graph structure
  optimization and constrained prediction. Mixed integer layers are not
  differentiable, so any connection would sit outside the training loop
  (for example, choosing which edges to add under a budget after
  training). No concrete graph learning use was strong enough for this
  release.
- **z3r.** z3r gives access to the Z3 solver. A natural use is to prove
  properties of small split specifications, such as “no held out pair
  appears in the training graph”. z3r needs Z3 4.8.13 or later and the
  Ubuntu releases used by the continuous integration runners ship
  4.8.12, so a bridge could not be tested there; the leakage properties
  are instead checked directly in the test suite.

Two other integrations in the brief were also looked at:

- **luz.** luz drives a training loop over dataloaders of tensors.
  graphlearnR’s loops need graph specific steps that luz does not model:
  sampled batches whose loss covers only seed nodes, link splits that
  rebuild negatives each epoch, and auxiliary losses such as DiffPool’s.
  Wrapping these as luz callbacks is possible but would duplicate
  [`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md);
  deferred until there is a concrete need.
- **tidymodels.** parsnip models expect one row per observation, while
  graph models need the whole graph at prediction time.
  [`predict()`](https://rdrr.io/r/stats/predict.html) already returns
  tidy data frames with `.pred_class` and `.prob_` columns, so yardstick
  metrics work on its output; a parsnip engine is deferred.
