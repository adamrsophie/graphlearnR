# Sibling packages and other frameworks

``` r

library(graphlearnR)
```

graphlearnR is one of a series of R packages by the same author. The
graph learning engine depends only on `torch`, `igraph` and `Matrix`;
every connection to a sibling package is optional, lives in `Suggests`,
and fails with a clear message when the sibling is not installed. This
article shows one bridge per package (modelcompR, symcompR, z3r,
optcompR, jaxR) and the connections to luz and tidymodels.

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

## Rule constrained labels with z3r

[z3r](https://github.com/adamrsophie/z3r) gives access to the Z3 solver.
[`consistent_labels()`](https://adamrsophie.github.io/graphlearnR/reference/consistent_labels.md)
takes per node class probabilities and logical rules on edges, such as
“two adjacent nodes cannot both be in class a”, and finds the most
probable labeling that obeys every rule with Z3’s optimizer. Nodes with
known labels can be fixed.

``` r

p <- rbind(c(a = 0.9, b = 0.1), c(a = 0.6, b = 0.4), c(a = 0.7, b = 0.3), c(a = 0.2, b = 0.8))
ring <- rbind(c(1, 2, 2, 3, 3, 4, 4, 1), c(2, 1, 3, 2, 4, 3, 1, 4))
res <- consistent_labels(p, ring, forbidden = cbind("a", "a"))
res$labels
res$changed
```

Independent predictions would label nodes 1, 2 and 3 as a; the rule
allows no two neighbors in a, and the cheapest fix in log probability
changes node 2. This runs after training on small graphs; it is a
combinatorial search, not a differentiable layer. The continuous
integration runners ship a Z3 release older than z3r needs, so this
bridge is tested on machines with a newer Z3 only.

## Budgeted link selection with optcompR

[optcompR](https://github.com/adamrsophie/optcompR) builds and solves
optimization models.
[`select_links()`](https://adamrsophie.github.io/graphlearnR/reference/select_links.md)
chooses which predicted links to add under a budget and a cap on new
links per node, maximizing the total score: a 0-1 integer program solved
with HiGHS.

``` r

sbm <- dataset_sbm(c(30, 30), seed = 1)
split <- split_edges(sbm, seed = 1)
torch::torch_manual_seed(1)
fit_link <- train_graph_model(gnn_model("gcn", ncol(sbm$x), 16), split, "link_prediction",
                              epochs = 50, seed = 1)
cand <- split$test_pos
prob <- predict(fit_link, pairs = cand, type = "prob")
pick <- select_links(cand, prob, budget = 5, max_new_degree = 1)
pick$pairs
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]   26   28   23   33   42
#> [2,]   30   31   32   43   58
round(prob[pick$selected], 3)
#> [1] 0.500 0.500 0.500 0.579 0.650
```

Integer programs are not differentiable, so this is a decision step
after training, as the design brief requires.

## A second array backend with jaxR

[jaxR](https://github.com/adamrsophie/jaxR) brings JAX’s transformations
to R.
[`aggregate_jax()`](https://adamrsophie.github.io/graphlearnR/reference/aggregate_jax.md)
is the neighborhood aggregation of
[`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md)
written in jaxR array operations, so it composes with `jaxR::grad()`,
`jaxR::vmap()` and `jaxR::jit()`. The tests check its values and
gradients against the torch version.

``` r

g_small <- graph_data(x = matrix(1:8, 4, 2), edge_index = rbind(c(1, 2, 3, 4), c(2, 3, 4, 2)),
                      num_nodes = 4)
as.matrix(aggregate_jax(g_small$x, g_small$edge_index, aggr = "mean"))
f <- function(x) jaxR::jnp_sum(aggregate_jax(x, g_small$edge_index, num_nodes = 4L)^2)
as.matrix(jaxR::grad(f)(jaxR::as_jax(g_small$x)))
```

Layers and training stay on torch; models are not converted between
backends. jaxR needs its XLA runtime (`jaxR::install_pjrt()`).

## Training with luz

[luz](https://mlverse.github.io/luz/) is the high level training library
of the torch ecosystem.
[`luz_node_dataloader()`](https://adamrsophie.github.io/graphlearnR/reference/luz_node_dataloader.md)
and `luz_node_module` adapt a node classification problem to it: the
dataloader yields the full graph with the indices of the masked nodes,
and the module returns predictions for those nodes only.

``` r

library(luz)
gk <- split_nodes(dataset_karate(), seed = 1)
torch::torch_manual_seed(1)
fitted <- luz_node_module |>
  setup(loss = torch::nn_cross_entropy_loss(), optimizer = torch::optim_adam,
        metrics = list(luz_metric_accuracy())) |>
  set_hparams(model = gnn_model("gcn", 34, 16, 2)) |>
  set_opt_hparams(lr = 0.01) |>
  fit(luz_node_dataloader(gk, "train"), epochs = 50, verbose = FALSE,
      valid_data = luz_node_dataloader(gk, "val"),
      accelerator = accelerator(cpu = TRUE))
unlist(tail(fitted$records$metrics$valid, 1))
#>       loss        acc 
#> 0.05139435 1.00000000
```

[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
remains the main trainer: it also covers sampled batches, link, edge and
graph tasks, and records a reproducibility trail.

## Tidy summaries

`augment()` and `glance()` from the generics package, the verbs used
across tidymodels, work on a `graph_fit`. `augment()` adds the true
outcome and the split of each node to the predictions, ready for
yardstick.

``` r

fit_k <- train_graph_model(gnn_model("gcn", 34, 8, 2), gk, "node_classification",
                           epochs = 30, seed = 1)
aug <- generics::augment(fit_k)
head(aug)
#>   .row node_id .pred_class .prob_Mr. Hi .prob_Officer .truth .split
#> 1    1       1      Mr. Hi    0.7682219     0.2317781 Mr. Hi  train
#> 2    2       2      Mr. Hi    0.7086080     0.2913920 Mr. Hi  train
#> 3    3       3      Mr. Hi    0.5500913     0.4499087 Mr. Hi  train
#> 4    4       4      Mr. Hi    0.6728596     0.3271404 Mr. Hi  train
#> 5    5       5      Mr. Hi    0.6685206     0.3314795 Mr. Hi    val
#> 6    6       6      Mr. Hi    0.6909202     0.3090798 Mr. Hi    val
with(aug[aug$.split == "test", ], mean(.pred_class == .truth))
#> [1] 0.8333333
generics::glance(fit_k)
#>                  task epochs best_epoch train_loss best_val_loss parameters
#> 1 node_classification     30         30  0.5332918     0.4581232        370
```

Graph models need the whole graph at prediction time, so there is no
parsnip engine.
