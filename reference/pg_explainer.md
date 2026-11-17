# Parameterized explainer (PGExplainer)

PGExplainer (Luo et al. 2020) learns one small network that predicts an
edge mask for **any** instance from the model's node embeddings, instead
of optimizing a separate mask per instance as
[`gnn_explainer()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
does. For an edge \\u \to v\\ the mask logit is \\\omega\_{uv} =
\mathrm{MLP}(\[z_u, z_v, z_t\])\\ for a node target \\t\\ (or \\\[z_u,
z_v\]\\ for a graph level output), with \\z\\ the trained model's last
hidden representation (`return_embedding = TRUE`), which is not updated.

## Usage

``` r
pg_explainer(
  model,
  data,
  targets = NULL,
  hidden = 64L,
  epochs = 30L,
  lr = 0.003,
  edge_size = 0.05,
  entropy = 1,
  temperature = c(5, 2),
  seed = NULL
)

# S3 method for class 'pg_explainer'
predict(object, g, target = NULL, ...)
```

## Arguments

- model:

  A trained model or `graph_fit` that accepts `return_embedding = TRUE`
  (for example from
  [`gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/gnn_model.md)).

- data:

  A `graph_data` object (node level models) or a list of them (graph
  level models).

- targets:

  For node level models, the target nodes to train on (default: training
  nodes, at most 100).

- hidden:

  MLP hidden size.

- epochs:

  Passes over the instances.

- lr:

  Learning rate.

- edge_size, entropy:

  Mask size and entropy penalties.

- temperature:

  Start and end temperature of the concrete relaxation.

- seed:

  Optional integer seed.

- object:

  A `pg_explainer`.

- g:

  Graph to explain.

- target:

  Target node (node level models), or `NULL`.

- ...:

  Unused.

## Value

`pg_explainer()`: an object of class `pg_explainer`;
`predict(explainer, g, target)`: a `graph_explanation` whose `edges`
table holds the mask probabilities.

## Details

Training samples a relaxed Bernoulli (concrete) mask \\m = \sigma((\log
\epsilon - \log(1 - \epsilon) + \omega) / \tau)\\, \\\epsilon \sim U(0,
1)\\, multiplies it into `edge_weight`, and minimizes the negative log
probability of the class the model predicts on the full graph (squared
change for one output regression), plus `edge_size` times the mask sum
and `entropy` times the mask entropy. The temperature \\\tau\\ decays
geometrically from `temperature[1]` to `temperature[2]`. The fitted
explainer then explains a new instance with one forward pass, \\m =
\sigma(\omega)\\.

Like
[`gnn_explainer()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md),
the mask acts through `edge_weight`, so the model must use edge weights;
the explanation shows how the model responds to edges, not causal
relevance.

## References

Luo et al. (2020) Parameterized explainer for graph neural network.
NeurIPS.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
                           "node_classification", epochs = 20, seed = 1)
  pg <- pg_explainer(fit, g, epochs = 3, seed = 1)
  ex <- predict(pg, g, target = 1)
  head(ex$edges[order(-ex$edges$attribution), ])
}
#>    edge from to attribution
#> 68   68   13  4  0.01748312
#> 67   67    4 13  0.01694749
#> 20   20   12  1  0.01660173
#> 66   66    8  4  0.01557513
#> 65   65    4  8  0.01512281
#> 70   70   14  4  0.01453398
```
