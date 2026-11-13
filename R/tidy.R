#' Tidy summaries of a fitted graph model
#'
#' Methods for the `generics` verbs used across tidymodels, so a
#' `graph_fit` can be summarized like other fitted models and its
#' predictions scored with yardstick.
#'
#' * `augment()` returns [predict()]'s data frame for node and graph tasks
#'   with the true outcome in `.truth` and a `.split` column (`"train"`,
#'   `"val"`, `"test"` or `NA`) for node tasks, ready for yardstick, for
#'   example `yardstick::accuracy(aug, .truth, .pred_class)`.
#' * `glance()` returns one row: task, epochs run, best epoch, final training
#'   loss, best monitored validation loss, and number of parameters.
#'
#' Graph models need the whole graph at prediction time, so there is no
#' parsnip engine; these methods cover the summary and evaluation steps.
#'
#' @param x A `graph_fit`.
#' @param new_data Data to predict on, as for [predict()].
#' @param ... Passed to [predict()].
#' @return A data frame (`augment()`) or a one row data frame (`glance()`).
#' @examples
#' if (torch::torch_is_installed() && requireNamespace("generics", quietly = TRUE)) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
#'                            "node_classification", epochs = 20, seed = 1)
#'   head(generics::augment(fit))
#'   generics::glance(fit)
#' }
#' @name graph_fit-tidy
#' @exportS3Method generics::augment
augment.graph_fit <- function(x, new_data = NULL, ...) {
  task <- x$task
  if (task %in% c("node_classification", "node_regression")) {
    g <- new_data %||% x$data
    if (!inherits(g, "graph_data")) stopf("`augment()` supports homogeneous node tasks and graph tasks.")
    out <- predict(x, g, ...)
    out$.truth <- g$y
    sp <- rep(NA_character_, g$num_nodes)
    for (k in c("train", "val", "test")) {
      m <- g[[paste0(k, "_mask")]]
      if (!is.null(m)) sp[m] <- k
    }
    out$.split <- sp
    return(out)
  }
  if (task %in% c("graph_classification", "graph_regression")) {
    data <- new_data %||% x$data
    if (inherits(data, "graph_data")) data <- list(data)
    out <- predict(x, data, ...)
    truth <- lapply(seq_along(data), function(i) data[[i]]$graph_y)
    out$.truth <- if (all(vapply(truth, is.factor, TRUE))) {
      factor(vapply(truth, as.character, ""), levels = levels(truth[[1L]]))
    } else {
      unlist(truth)
    }
    return(out)
  }
  stopf("`augment()` supports node and graph tasks; use evaluate_graph_model() for %s.", task)
}

#' @rdname graph_fit-tidy
#' @exportS3Method generics::glance
glance.graph_fit <- function(x, ...) {
  h <- x$history
  vl <- h$val_loss[!is.na(h$val_loss)]
  data.frame(task = x$task, epochs = nrow(h), best_epoch = x$best_epoch %||% NA_integer_,
             train_loss = utils::tail(h$train_loss, 1L),
             best_val_loss = if (length(vl)) min(vl) else NA_real_,
             parameters = sum(vapply(x$model$parameters, function(p) p$numel(), numeric(1))))
}
