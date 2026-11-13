#' Evaluation metrics
#'
#' Task metrics computed on plain R vectors, so they can be used with any
#' model. Classification metrics take the true class and the predicted class;
#' ranking metrics take scores (higher means more likely positive).
#'
#' * `metric_roc_auc()`: area under the ROC curve via the Mann-Whitney
#'   statistic, with ties counted one half. For a probability matrix and
#'   factor truth with more than two levels it returns the macro average of
#'   one-vs-rest AUCs.
#' * `metric_average_precision()`: \eqn{\sum_n (R_n - R_{n-1}) P_n} over
#'   distinct score thresholds (the definition used by scikit-learn), not
#'   the trapezoidal area under the precision-recall curve. A probability
#'   matrix gives the macro one-vs-rest average, as for ROC-AUC.
#' * `metric_ece()`: expected calibration error with equal width bins on the
#'   top predicted probability.
#' * `metric_hits_at_k()`, `metric_mrr()`: link ranking metrics. With a
#'   vector of shared negatives, Hits@K is the fraction of positives scored
#'   above the K-th highest negative (the OGB `ogbl-collab` convention).
#'   With a matrix of per positive negatives (one row per positive), the rank
#'   of each positive among its own negatives is used, ties counted one half.
#'
#' @param truth True values: factor or integer classes, 0/1 or logical for
#'   binary ranking metrics, numeric for regression.
#' @param estimate Predicted class or value.
#' @param score Predicted score or probability of the positive class (or a
#'   probability matrix with one column per class for `metric_roc_auc()`,
#'   `metric_average_precision()` and `metric_ece()`).
#' @param average `"macro"` or `"micro"` averaging over classes, or
#'   `"binary"` for the class named by `positive`.
#' @param positive Positive class for binary averaging.
#' @return A numeric scalar.
#' @examples
#' truth <- factor(c("a", "b", "a", "b"))
#' est <- factor(c("a", "b", "b", "b"))
#' metric_accuracy(truth, est)
#' metric_f1(truth, est)
#' metric_roc_auc(c(1, 0, 1, 0), c(0.9, 0.2, 0.4, 0.5))
#' metric_hits_at_k(c(0.9, 0.3), c(0.5, 0.4, 0.1), k = 1)
#' @name metrics
NULL

align_classes <- function(truth, estimate) {
  lv <- union(levels(as.factor(truth)), levels(as.factor(estimate)))
  list(t = factor(as.character(truth), levels = lv),
       e = factor(as.character(estimate), levels = lv))
}

#' @rdname metrics
#' @export
metric_accuracy <- function(truth, estimate) {
  mean(as.character(truth) == as.character(estimate))
}

#' @rdname metrics
#' @export
metric_balanced_accuracy <- function(truth, estimate) {
  a <- align_classes(truth, estimate)
  present <- levels(a$t)[table(a$t) > 0]
  mean(vapply(present, function(l) mean(a$e[a$t == l] == l), numeric(1)))
}

confusion_counts <- function(truth, estimate) {
  a <- align_classes(truth, estimate)
  lv <- levels(a$t)
  tp <- vapply(lv, function(l) sum(a$t == l & a$e == l), numeric(1))
  fp <- vapply(lv, function(l) sum(a$t != l & a$e == l), numeric(1))
  fn <- vapply(lv, function(l) sum(a$t == l & a$e != l), numeric(1))
  list(tp = tp, fp = fp, fn = fn, present = table(a$t) > 0 | table(a$e) > 0)
}

safe_div <- function(a, b) ifelse(b == 0, 0, a / b)

prf <- function(truth, estimate, average, positive, what) {
  average <- match.arg(average, c("macro", "micro", "binary"))
  cc <- confusion_counts(truth, estimate)
  if (average == "micro") {
    cc <- lapply(cc[1:3], sum)
  } else if (average == "binary") {
    positive <- positive %||% levels(align_classes(truth, estimate)$t)[2L]
    cc <- lapply(cc[1:3], function(v) v[[as.character(positive)]])
  }
  p <- safe_div(cc$tp, cc$tp + cc$fp)
  r <- safe_div(cc$tp, cc$tp + cc$fn)
  f <- safe_div(2 * p * r, p + r)
  v <- switch(what, precision = p, recall = r, f1 = f)
  if (average == "macro") mean(v[cc$present]) else v
}

#' @rdname metrics
#' @export
metric_precision <- function(truth, estimate, average = "macro", positive = NULL) {
  prf(truth, estimate, average, positive, "precision")
}

#' @rdname metrics
#' @export
metric_recall <- function(truth, estimate, average = "macro", positive = NULL) {
  prf(truth, estimate, average, positive, "recall")
}

#' @rdname metrics
#' @export
metric_f1 <- function(truth, estimate, average = "macro", positive = NULL) {
  prf(truth, estimate, average, positive, "f1")
}

as_binary <- function(truth) {
  if (is.factor(truth)) {
    if (nlevels(truth) != 2L) stopf("Binary metrics need two classes.")
    return(as.integer(truth) == 2L)
  }
  as.logical(truth)
}

#' @rdname metrics
#' @export
metric_roc_auc <- function(truth, score) {
  if (is.matrix(score)) {
    truth <- as.factor(truth)
    if (ncol(score) == 2L && nlevels(truth) == 2L) {
      return(metric_roc_auc(as.integer(truth) == 2L, score[, 2L]))
    }
    aucs <- vapply(seq_len(ncol(score)), function(j) {
      y <- as.integer(truth) == j
      if (all(y) || !any(y)) NA_real_ else metric_roc_auc(y, score[, j])
    }, numeric(1))
    return(mean(aucs, na.rm = TRUE))
  }
  y <- as_binary(truth)
  n1 <- sum(y)
  n0 <- sum(!y)
  if (n1 == 0 || n0 == 0) stopf("ROC-AUC needs both positive and negative cases.")
  r <- rank(score)
  (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

#' @rdname metrics
#' @export
metric_average_precision <- function(truth, score) {
  if (is.matrix(score)) {
    # Binary: the second column scores the positive class. More classes:
    # macro average of one versus rest average precision over the classes
    # that occur in `truth`.
    truth <- as.factor(truth)
    if (ncol(score) == 2L && nlevels(truth) == 2L) {
      return(metric_average_precision(as.integer(truth) == 2L, score[, 2L]))
    }
    aps <- vapply(seq_len(ncol(score)), function(j) {
      y <- as.integer(truth) == j
      if (!any(y)) NA_real_ else metric_average_precision(y, score[, j])
    }, numeric(1))
    return(mean(aps, na.rm = TRUE))
  }
  y <- as_binary(truth)
  if (!any(y)) stopf("Average precision needs at least one positive case.")
  o <- order(-score)
  y <- y[o]
  s <- score[o]
  last <- c(s[-1L] != s[-length(s)], TRUE)
  tp <- cumsum(y)[last]
  k <- seq_along(y)[last]
  prec <- tp / k
  rec <- tp / sum(y)
  sum(diff(c(0, rec)) * prec)
}

#' @rdname metrics
#' @param prob Probability matrix (one column per class) or vector of
#'   positive class probabilities.
#' @param bins Number of equal width bins.
#' @export
metric_ece <- function(truth, prob, bins = 10L) {
  if (!is.matrix(prob)) prob <- cbind(1 - prob, prob)
  truth <- as.integer(as.factor(truth))
  conf <- apply(prob, 1L, max)
  pred <- max.col(prob, ties.method = "first")
  bin <- pmin(floor(conf * bins) + 1L, bins)
  acc <- tapply(pred == truth, bin, mean)
  cf <- tapply(conf, bin, mean)
  w <- tapply(conf, bin, length) / length(conf)
  sum(w * abs(acc - cf))
}

#' @rdname metrics
#' @export
metric_brier <- function(truth, prob) {
  if (!is.matrix(prob)) {
    return(mean((prob - as_binary(truth))^2))
  }
  y <- stats::model.matrix(~ f - 1, data.frame(f = factor(as.integer(as.factor(truth)),
                                                          levels = seq_len(ncol(prob)))))
  mean(rowSums((prob - y)^2))
}

#' @rdname metrics
#' @export
metric_mse <- function(truth, estimate) {
  check_same_length(truth, estimate)
  mean((truth - estimate)^2)
}

check_same_length <- function(truth, estimate) {
  if (length(truth) != length(estimate)) {
    stopf("`truth` has %d values but `estimate` has %d.", length(truth), length(estimate))
  }
}

#' @rdname metrics
#' @export
metric_rmse <- function(truth, estimate) sqrt(metric_mse(truth, estimate))

#' @rdname metrics
#' @export
metric_mae <- function(truth, estimate) {
  check_same_length(truth, estimate)
  mean(abs(truth - estimate))
}

#' @rdname metrics
#' @export
metric_r2 <- function(truth, estimate) {
  check_same_length(truth, estimate)
  1 - sum((truth - estimate)^2) / sum((truth - mean(truth))^2)
}

positive_ranks <- function(pos, neg) {
  if (!is.matrix(neg) || nrow(neg) != length(pos)) {
    stopf("`neg` must be a matrix with one row per positive score.")
  }
  1 + rowSums(neg > pos) + 0.5 * rowSums(neg == pos)
}

#' @rdname metrics
#' @param pos Scores of positive pairs.
#' @param neg Scores of negative pairs: a vector shared by all positives, or
#'   a matrix with one row per positive.
#' @param k Cut-off.
#' @export
metric_hits_at_k <- function(pos, neg, k = 10L) {
  if (is.matrix(neg)) {
    return(mean(positive_ranks(pos, neg) <= k))
  }
  if (length(neg) < k) return(1)
  kth <- sort(neg, decreasing = TRUE)[k]
  mean(pos > kth)
}

#' @rdname metrics
#' @export
metric_mrr <- function(pos, neg) {
  if (!is.matrix(neg)) neg <- matrix(neg, length(pos), length(neg), byrow = TRUE)
  mean(1 / positive_ranks(pos, neg))
}

#' @rdname metrics
#' @param cluster Cluster assignment.
#' @export
metric_nmi <- function(truth, cluster) {
  tab <- table(truth, cluster)
  p <- tab / sum(tab)
  pi <- rowSums(p)
  pj <- colSums(p)
  nz <- p > 0
  mi <- sum(p[nz] * log(p[nz] / outer(pi, pj)[nz]))
  h <- function(q) -sum(q[q > 0] * log(q[q > 0]))
  denom <- (h(pi) + h(pj)) / 2
  if (denom == 0) 1 else mi / denom
}

#' @rdname metrics
#' @export
metric_ari <- function(truth, cluster) {
  tab <- table(truth, cluster)
  c2 <- function(x) x * (x - 1) / 2
  s <- sum(c2(tab))
  a <- sum(c2(rowSums(tab)))
  b <- sum(c2(colSums(tab)))
  e <- a * b / c2(sum(tab))
  m <- (a + b) / 2
  if (m == e) 1 else (s - e) / (m - e)
}

#' Evaluate predictions for a task
#'
#' Computes a standard metric set for a task from truth and predictions.
#'
#' @param truth True labels or values.
#' @param prediction For classification, a probability matrix (one column
#'   per class, column names as class labels) or a vector of positive class
#'   probabilities; for regression, predicted values; for link prediction,
#'   positive scores.
#' @param task One of `"classification"`, `"regression"`, `"link"`.
#' @param neg Negative scores for `"link"`.
#' @param ks Cut-offs for Hits@K.
#' @return A data frame with columns `metric` and `value`.
#' @examples
#' p <- matrix(c(0.8, 0.2, 0.3, 0.7, 0.6, 0.4), ncol = 2, byrow = TRUE,
#'             dimnames = list(NULL, c("a", "b")))
#' evaluate_predictions(factor(c("a", "b", "b")), p, "classification")
#' @export
evaluate_predictions <- function(truth, prediction, task = c("classification", "regression", "link"),
                                 neg = NULL, ks = c(1L, 10L, 50L)) {
  task <- match.arg(task)
  out <- switch(task,
    classification = {
      prob <- if (is.matrix(prediction)) prediction else cbind(1 - prediction, prediction)
      lv <- colnames(prob) %||% levels(as.factor(truth))
      est <- factor(lv[max.col(prob, ties.method = "first")], levels = lv)
      tf <- factor(as.character(truth), levels = lv)
      m <- c(accuracy = metric_accuracy(tf, est),
             balanced_accuracy = metric_balanced_accuracy(tf, est),
             precision_macro = metric_precision(tf, est),
             recall_macro = metric_recall(tf, est),
             f1_macro = metric_f1(tf, est),
             ece = metric_ece(tf, prob),
             brier = metric_brier(tf, prob))
      auc <- tryCatch(metric_roc_auc(tf, prob), error = function(e) NA_real_)
      ap <- tryCatch(metric_average_precision(tf, prob), error = function(e) NA_real_)
      c(m, roc_auc = auc, average_precision = ap)
    },
    regression = c(mse = metric_mse(truth, prediction),
                   rmse = metric_rmse(truth, prediction),
                   mae = metric_mae(truth, prediction),
                   r2 = metric_r2(truth, prediction)),
    link = {
      if (is.null(neg)) stopf("Link evaluation needs negative scores `neg`.")
      y <- c(rep(TRUE, length(prediction)), rep(FALSE, length(neg)))
      s <- c(prediction, as.vector(neg))
      hk <- vapply(ks, function(k) metric_hits_at_k(prediction, neg, k), numeric(1))
      c(roc_auc = metric_roc_auc(y, s), average_precision = metric_average_precision(y, s),
        mrr = metric_mrr(prediction, neg), setNames(hk, paste0("hits@", ks)))
    }
  )
  data.frame(metric = names(out), value = unname(out), row.names = NULL)
}

#' Evaluate node or graph embeddings
#'
#' Two task independent checks of a learned representation, for example
#' `predict(fit, type = "embedding")`:
#'
#' * **Downstream prediction (linear probe).** A ridge regression from the
#'   standardized embedding to one hot labels (classification) or to the
#'   numeric target is fitted on the `train` rows and scored on the others.
#'   Classification reports accuracy and macro F1; regression reports RMSE
#'   and \eqn{R^2}. A linear probe measures how much label information is
#'   linearly available, not the best attainable model.
#' * **Clustering.** k-means with `k` centers (default: the number of
#'   classes) on the standardized embedding, compared with the labels by
#'   normalized mutual information and the adjusted Rand index.
#'
#' The split of `train` rows must follow the dependence structure of the
#' data (for example [split_nodes()] masks), not a random mix of dependent
#' observations.
#'
#' @param embedding Numeric matrix, one row per node or graph.
#' @param labels Factor (classification) or numeric vector, one per row.
#' @param train Logical vector of rows used to fit the probe.
#' @param lambda Ridge penalty.
#' @param k Number of clusters; `NULL` uses the number of classes. Ignored
#'   (no clustering) for numeric labels unless given.
#' @param seed Optional seed for k-means starts.
#' @return A data frame with columns `metric` and `value`.
#' @examples
#' g <- dataset_sbm(c(30, 30), p_in = 0.3, p_out = 0.02, seed = 1)
#' emb <- laplacian_pe(g, 4)
#' tr <- rep(c(TRUE, FALSE), 30)
#' evaluate_embeddings(emb, g$y, train = tr, seed = 1)
#' @export
evaluate_embeddings <- function(embedding, labels, train, lambda = 1e-3, k = NULL, seed = NULL) {
  emb <- as.matrix(embedding)
  if (!is.numeric(emb) || anyNA(emb)) stopf("`embedding` must be a numeric matrix without missing values.")
  n <- nrow(emb)
  if (length(labels) != n) stopf("`labels` has %d entries for %d embedding rows.", length(labels), n)
  train <- as.logical(train)
  if (length(train) != n || anyNA(train)) stopf("`train` must be a logical vector with one entry per row.")
  if (!any(train) || all(train)) stopf("`train` must leave rows on both sides.")
  mu <- colMeans(emb[train, , drop = FALSE])
  sdv <- apply(emb[train, , drop = FALSE], 2L, stats::sd)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  z <- sweep(sweep(emb, 2L, mu), 2L, sdv, "/")
  X <- cbind(1, z)
  classif <- is.factor(labels) || is.character(labels) || is.logical(labels)
  Y <- if (classif) {
    labels <- as.factor(labels)
    stats::model.matrix(~ labels - 1)
  } else {
    matrix(as.numeric(labels))
  }
  Xt <- X[train, , drop = FALSE]
  pen <- diag(lambda * sum(train), ncol(X))
  pen[1L, 1L] <- 0
  B <- solve(crossprod(Xt) + pen, crossprod(Xt, Y[train, , drop = FALSE]))
  P <- X[!train, , drop = FALSE] %*% B
  out <- if (classif) {
    est <- factor(levels(labels)[max.col(P, ties.method = "first")], levels = levels(labels))
    tr <- labels[!train]
    c(probe_accuracy = metric_accuracy(tr, est), probe_f1_macro = metric_f1(tr, est))
  } else {
    c(probe_rmse = metric_rmse(Y[!train, 1L], P[, 1L]), probe_r2 = metric_r2(Y[!train, 1L], P[, 1L]))
  }
  k <- k %||% if (classif) nlevels(labels)
  if (!is.null(k)) {
    cl <- with_optional_seed(seed, stats::kmeans(z, centers = k, nstart = 10L)$cluster)
    if (classif) out <- c(out, cluster_nmi = metric_nmi(labels, cl), cluster_ari = metric_ari(labels, cl))
  }
  data.frame(metric = names(out), value = unname(out), row.names = NULL)
}
