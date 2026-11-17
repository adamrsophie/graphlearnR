# Subgraph explanations by Monte Carlo tree search (SubgraphX)

SubgraphX (Yuan et al. 2021) searches for a small connected subgraph
that explains a prediction. Each node of the search tree is a node set;
its children remove one node and keep the connected component that holds
the target (node level) or the largest component (graph level). A
subgraph \\S\\ is scored by its Shapley value as one player among the
other nodes within `hops` hops of it, \$\$\phi(S) \approx
\frac{1}{M}\sum\_{s=1}^{M} f(C_s \cup S) - f(C_s),\$\$ with coalitions
\\C_s\\ drawn as the predecessors of \\S\\ in a random ordering (a
uniform size, then a uniform subset), and \\f\\ the explained output
when only the given nodes are kept: other nodes have their features set
to zero and lose their edges (the target node of a node level
explanation is always kept). Search follows the paper's upper confidence
rule, \\Q + c \\ \phi \sqrt{\sum N} / (1 + N)\\, over `rollouts`
descents, expanding at most `expand_atoms` children per node (the lowest
degree nodes first). The best scoring subgraph with at most `max_nodes`
nodes is returned.

## Usage

``` r
subgraphx(
  model,
  g,
  target = NULL,
  output = NULL,
  hops = 2L,
  max_nodes = 5L,
  rollouts = 10L,
  expand_atoms = 4L,
  samples = 10L,
  c_puct = 10,
  seed = NULL
)
```

## Arguments

- model:

  A trained model or `graph_fit`.

- g:

  A `graph_data` object.

- target:

  Target node, or `NULL` for a graph level output.

- output:

  Output column; default the predicted class (or the single regression
  output).

- hops:

  Neighborhood radius: the model's number of message passing layers.

- max_nodes:

  Largest explanation size.

- rollouts:

  Tree search descents.

- expand_atoms:

  Children expanded per tree node.

- samples:

  Monte Carlo samples per Shapley value.

- c_puct:

  Exploration constant.

- seed:

  Optional integer seed.

## Value

A `graph_explanation` with `subgraph_nodes`, its Shapley `score`, a
`candidates` table of evaluated subgraphs, and an `edges` table with
attribution 1 for edges inside the subgraph and 0 otherwise.

## Details

Removing nodes rather than masking edges works with any model, including
attention models that ignore `edge_weight`. The cost is many forward
passes (about `rollouts` times the search depth times `expand_atoms`
times `samples`, with caching), so this suits small graphs and
neighborhoods.

## References

Yuan, Yu, Wang, Li and Ji (2021) On explainability of graph neural
networks via subgraph explorations. ICML.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
                           "node_classification", epochs = 20, seed = 1)
  ex <- subgraphx(fit, g, target = 1, max_nodes = 4, rollouts = 3, samples = 5, seed = 1)
  ex$subgraph_nodes
}
#> [1]  1  3  4 14
```
