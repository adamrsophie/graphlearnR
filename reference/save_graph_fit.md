# Save and load a trained model

`save_graph_fit()` writes the model and decoder with `torch`'s own
serialization
([`torch::torch_serialize()`](https://torch.mlverse.org/docs/reference/torch_serialize.html))
together with the task, class levels, training history and
reproducibility record into one `.rds` file. Training data and optimizer
state are not saved. `load_graph_fit()` restores a fit that
[`predict()`](https://rdrr.io/r/stats/predict.html) can use.

## Usage

``` r
save_graph_fit(fit, path)

load_graph_fit(path, device = NULL)
```

## Arguments

- fit:

  A `graph_fit`.

- path:

  File path.

- device:

  Device to load onto.

## Value

`save_graph_fit()` returns `path` invisibly; `load_graph_fit()` returns
a `graph_fit` (without `data` and `optimizer`).

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification",
                           epochs = 5, seed = 1)
  path <- tempfile(fileext = ".rds")
  save_graph_fit(fit, path)
  back <- load_graph_fit(path)
  all.equal(predict(fit, g, type = "prob"), predict(back, g, type = "prob"))
}
#> [1] TRUE
```
