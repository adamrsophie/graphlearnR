# Evaluate predictions for a task

Computes a standard metric set for a task from truth and predictions.

## Usage

``` r
evaluate_predictions(
  truth,
  prediction,
  task = c("classification", "regression", "link"),
  neg = NULL,
  ks = c(1L, 10L, 50L)
)
```

## Arguments

- truth:

  True labels or values.

- prediction:

  For classification, a probability matrix (one column per class, column
  names as class labels) or a vector of positive class probabilities;
  for regression, predicted values; for link prediction, positive
  scores.

- task:

  One of `"classification"`, `"regression"`, `"link"`.

- neg:

  Negative scores for `"link"`.

- ks:

  Cut-offs for Hits@K.

## Value

A data frame with columns `metric` and `value`.

## Examples

``` r
p <- matrix(c(0.8, 0.2, 0.3, 0.7, 0.6, 0.4), ncol = 2, byrow = TRUE,
            dimnames = list(NULL, c("a", "b")))
evaluate_predictions(factor(c("a", "b", "b")), p, "classification")
#>              metric     value
#> 1          accuracy 0.6666667
#> 2 balanced_accuracy 0.7500000
#> 3   precision_macro 0.7500000
#> 4      recall_macro 0.7500000
#> 5          f1_macro 0.6666667
#> 6               ece 0.3666667
#> 7             brier 0.3266667
#> 8           roc_auc 1.0000000
```
