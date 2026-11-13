# Evaluation metrics

Task metrics computed on plain R vectors, so they can be used with any
model. Classification metrics take the true class and the predicted
class; ranking metrics take scores (higher means more likely positive).

## Usage

``` r
metric_accuracy(truth, estimate)

metric_balanced_accuracy(truth, estimate)

metric_precision(truth, estimate, average = "macro", positive = NULL)

metric_recall(truth, estimate, average = "macro", positive = NULL)

metric_f1(truth, estimate, average = "macro", positive = NULL)

metric_roc_auc(truth, score)

metric_average_precision(truth, score)

metric_ece(truth, prob, bins = 10L)

metric_brier(truth, prob)

metric_mse(truth, estimate)

metric_rmse(truth, estimate)

metric_mae(truth, estimate)

metric_r2(truth, estimate)

metric_hits_at_k(pos, neg, k = 10L)

metric_mrr(pos, neg)

metric_nmi(truth, cluster)

metric_ari(truth, cluster)
```

## Arguments

- truth:

  True values: factor or integer classes, 0/1 or logical for binary
  ranking metrics, numeric for regression.

- estimate:

  Predicted class or value.

- average:

  `"macro"` or `"micro"` averaging over classes, or `"binary"` for the
  class named by `positive`.

- positive:

  Positive class for binary averaging.

- score:

  Predicted score or probability of the positive class (or a probability
  matrix with one column per class for `metric_roc_auc()` and
  `metric_ece()`).

- prob:

  Probability matrix (one column per class) or vector of positive class
  probabilities.

- bins:

  Number of equal width bins.

- pos:

  Scores of positive pairs.

- neg:

  Scores of negative pairs: a vector shared by all positives, or a
  matrix with one row per positive.

- k:

  Cut-off.

- cluster:

  Cluster assignment.

## Value

A numeric scalar.

## Details

- `metric_roc_auc()`: area under the ROC curve via the Mann-Whitney
  statistic, with ties counted one half. For a probability matrix and
  factor truth with more than two levels it returns the macro average of
  one-vs-rest AUCs.

- `metric_average_precision()`: \\\sum_n (R_n - R\_{n-1}) P_n\\ over
  distinct score thresholds (the definition used by scikit-learn), not
  the trapezoidal area under the precision-recall curve.

- `metric_ece()`: expected calibration error with equal width bins on
  the top predicted probability.

- `metric_hits_at_k()`, `metric_mrr()`: link ranking metrics. With a
  vector of shared negatives, Hits@K is the fraction of positives scored
  above the K-th highest negative (the OGB `ogbl-collab` convention).
  With a matrix of per positive negatives (one row per positive), the
  rank of each positive among its own negatives is used, ties counted
  one half.

## Examples

``` r
truth <- factor(c("a", "b", "a", "b"))
est <- factor(c("a", "b", "b", "b"))
metric_accuracy(truth, est)
#> [1] 0.75
metric_f1(truth, est)
#> [1] 0.7333333
metric_roc_auc(c(1, 0, 1, 0), c(0.9, 0.2, 0.4, 0.5))
#> [1] 0.75
metric_hits_at_k(c(0.9, 0.3), c(0.5, 0.4, 0.1), k = 1)
#> [1] 0.5
```
