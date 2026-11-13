# Training node models with luz

Helpers to train a node level graph model with
[luz](https://mlverse.github.io/luz/) instead of
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md).
`luz_node_dataloader()` returns a dataloader with a single full graph
batch whose input is `list(x, edge_index, idx)` (plus `edge_weight` when
the graph has one) and whose target is the labels of the nodes in
`mask`; `luz_node_module` wraps any model whose forward takes
`(x, edge_index, edge_weight = )` so that it returns predictions for
those nodes only. Losses are therefore computed on the masked nodes, and
message passing still uses the whole graph.

## Usage

``` r
luz_node_dataloader(g, mask = c("train", "val", "test"))

luz_node_module(model)
```

## Arguments

- g:

  A `graph_data` object with node labels and masks.

- mask:

  `"train"`, `"val"` or `"test"`.

## Value

`luz_node_dataloader()`: a torch dataloader. `luz_node_module`: an
`nn_module` generator taking the wrapped `model`.

## Details

Use it with
[`luz::setup()`](https://mlverse.github.io/luz/reference/setup.html),
`luz::set_hparams(model = ...)` and
[`luz::fit()`](https://generics.r-lib.org/reference/fit.html); see the
example. luz handles devices, callbacks and metrics; graphlearnR's own
trainer adds neighbor and subgraph sampling, link and graph tasks, and a
reproducibility record, which this path does not.

## Examples

``` r
if (torch::torch_is_installed() && requireNamespace("luz", quietly = TRUE)) {
  library(luz)
  g <- split_nodes(dataset_karate(), seed = 1)
  fitted <- luz_node_module |>
    setup(loss = torch::nn_cross_entropy_loss(), optimizer = torch::optim_adam) |>
    set_hparams(model = gnn_model("gcn", 34, 16, 2)) |>
    fit(luz_node_dataloader(g, "train"), epochs = 20, verbose = FALSE,
        valid_data = luz_node_dataloader(g, "val"),
        accelerator = accelerator(cpu = TRUE))
}
```
