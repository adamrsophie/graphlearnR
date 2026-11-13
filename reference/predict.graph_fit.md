# Predict with a trained graph model

Runs the model in evaluation mode without gradients.

## Usage

``` r
# S3 method for class 'graph_fit'
predict(
  object,
  new_data = NULL,
  type = c("frame", "class", "prob", "response", "embedding", "score"),
  pairs = NULL,
  relation = NULL,
  batch_size = 64L,
  ...
)
```

## Arguments

- object:

  A `graph_fit`.

- new_data:

  Data to predict on; defaults to the training data where that makes
  sense.

- type:

  `"frame"` (a `graph_predictions` data frame), `"class"`, `"prob"`,
  `"response"` (raw model output), `"embedding"` (node embeddings; for
  graph tasks one pooled row per graph, using the model's readout or the
  mean) or `"score"` (link logits).

- pairs:

  2 by K matrix of node pairs (link and edge tasks).

- relation:

  Relation codes for multi relational decoders.

- batch_size:

  Graphs per batch for graph tasks.

- ...:

  Unused.

## Value

Depending on `type`; the default is a data frame of class
`graph_predictions` with `.row`, an identifier column, `.pred_class` and
`.prob_<level>` columns (classification), `.pred` (regression) or
`.score` and `.prob` (links).

## Details

- Node tasks: one row per node of `new_data` (a `graph_data`, or a
  `hetero_graph_data` for the fitted target type).

- Graph tasks: one row per graph of `new_data` (a list of graphs or a
  `graph_dataset`).

- Link prediction: one row per pair in `pairs`; message passing uses
  `new_data`, by default the training graph of the split.

- Edge tasks: one row per pair in `pairs` (default every stored edge).
  When the model uses edge features, each requested pair takes the
  features of its stored edge in `new_data`; pairs that are not edges
  are an error.

## Examples

``` r
if (torch::torch_is_installed()) {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification",
                           epochs = 20, seed = 1)
  head(predict(fit, g))
  head(predict(fit, g, type = "prob"))
}
#>         Mr. Hi   Officer
#> [1,] 0.5901923 0.4098077
#> [2,] 0.5661643 0.4338357
#> [3,] 0.4643112 0.5356889
#> [4,] 0.5628192 0.4371808
#> [5,] 0.5735875 0.4264125
#> [6,] 0.5763327 0.4236672
```
