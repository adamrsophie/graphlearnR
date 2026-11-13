# Tidy summaries of a fitted graph model

Methods for the `generics` verbs used across tidymodels, so a
`graph_fit` can be summarized like other fitted models and its
predictions scored with yardstick.

## Usage

``` r
# S3 method for class 'graph_fit'
augment(x, new_data = NULL, ...)

# S3 method for class 'graph_fit'
glance(x, ...)
```

## Arguments

- x:

  A `graph_fit`.

- new_data:

  Data to predict on, as for
  [`predict()`](https://rdrr.io/r/stats/predict.html).

- ...:

  Passed to [`predict()`](https://rdrr.io/r/stats/predict.html).

## Value

A data frame (`augment()`) or a one row data frame (`glance()`).

## Details

- `augment()` returns
  [`predict()`](https://rdrr.io/r/stats/predict.html)'s data frame for
  node and graph tasks with the true outcome in `.truth` and a `.split`
  column (`"train"`, `"val"`, `"test"` or `NA`) for node tasks, ready
  for yardstick, for example
  `yardstick::accuracy(aug, .truth, .pred_class)`.

- `glance()` returns one row: task, epochs run, best epoch, final
  training loss, best monitored validation loss, and number of
  parameters.

Graph models need the whole graph at prediction time, so there is no
parsnip engine; these methods cover the summary and evaluation steps.

## Examples

``` r
if (torch::torch_is_installed() && requireNamespace("generics", quietly = TRUE)) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
                           "node_classification", epochs = 20, seed = 1)
  head(generics::augment(fit))
  generics::glance(fit)
}
#>                  task epochs best_epoch train_loss best_val_loss parameters
#> 1 node_classification     20         20  0.6691499     0.6362948        370
```
