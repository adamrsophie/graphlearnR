#' Predict with a trained graph model
#'
#' Runs the model in evaluation mode without gradients.
#'
#' * Node tasks: one row per node of `new_data` (a `graph_data`, or a
#'   `hetero_graph_data` for the fitted target type).
#' * Graph tasks: one row per graph of `new_data` (a list of graphs or a
#'   `graph_dataset`).
#' * Link prediction: one row per pair in `pairs`; message passing uses
#'   `new_data`, by default the training graph of the split.
#' * Edge tasks: one row per pair in `pairs` (default every stored edge).
#'
#' @param object A `graph_fit`.
#' @param new_data Data to predict on; defaults to the training data where
#'   that makes sense.
#' @param type `"frame"` (a `graph_predictions` data frame), `"class"`,
#'   `"prob"`, `"response"` (raw model output), `"embedding"` (node
#'   embeddings; for graph tasks one pooled row per graph, using the model's
#'   readout or the mean) or `"score"` (link logits).
#' @param pairs 2 by K matrix of node pairs (link and edge tasks).
#' @param relation Relation codes for multi relational decoders.
#' @param batch_size Graphs per batch for graph tasks.
#' @param ... Unused.
#' @return Depending on `type`; the default is a data frame of class
#'   `graph_predictions` with `.row`, an identifier column, `.pred_class`
#'   and `.prob_<level>` columns (classification), `.pred` (regression) or
#'   `.score` and `.prob` (links).
#' @export
predict.graph_fit <- function(object, new_data = NULL,
                              type = c("frame", "class", "prob", "response",
                                       "embedding", "score"),
                              pairs = NULL, relation = NULL, batch_size = 64L, ...) {
  type <- match.arg(type)
  model <- object$model
  model$eval()
  dev <- object$device
  task <- object$task
  with_no_grad({
    if (task %in% c("node_classification", "node_regression")) {
      new_data <- new_data %||% object$data
      if (inherits(new_data, "hetero_graph_data")) {
        ht <- hetero_tensors(new_data, dev)
        out <- model(ht, return_embedding = type == "embedding")[[object$target_type]]
        ids <- new_data$nodes[[object$target_type]]$node_ids
      } else {
        out <- model_forward(graph_tensors(new_data, dev), model = model,
                             return_embedding = type == "embedding")
        ids <- new_data$node_ids
      }
      return(format_predictions(out, type, object, ids, "node_id"))
    }
    if (task %in% c("graph_classification", "graph_regression")) {
      new_data <- new_data %||% object$data
      if (inherits(new_data, "graph_data")) new_data <- list(new_data)
      batches <- loader_batches(graph_loader(new_data, batch_size, FALSE))
      out <- torch_cat(lapply(batches, function(b) {
        bt <- graph_tensors(b, dev)
        out <- model_forward(model, bt, return_embedding = type == "embedding")
        if (type == "embedding") out <- pool_graph_embedding(model, out, bt)
        out
      }), dim = 1L)
      return(format_predictions(out, type, object, seq_len(out$size(1L)), "graph"))
    }
    if (task == "link_prediction") {
      split <- object$data
      if (inherits(split, "hetero_link_split")) {
        return(predict_hetero_links(object, new_data, pairs, type))
      }
      g <- new_data %||% split$train_graph
      z <- model_forward(model, graph_tensors(g, dev))
      if (type == "embedding") return(as_r_array(z))
      if (is.null(pairs)) stopf("Link prediction needs `pairs` to score.")
      pairs <- normalize_edge_index(pairs)
      s <- object$decoder(z, torch_tensor(pairs, dtype = torch_long(), device = dev), relation)
      s <- as.vector(as_r_array(s))
      if (type == "score" || type == "response") return(s)
      if (type == "prob") return(stats::plogis(s))
      return(structure(data.frame(.row = seq_along(s), from = g$node_ids[pairs[1L, ]],
                                  to = g$node_ids[pairs[2L, ]], .score = s,
                                  .prob = stats::plogis(s)),
                       class = c("graph_predictions", "data.frame")))
    }
    g <- new_data %||% object$data
    pairs <- if (is.null(pairs)) g$edge_index else normalize_edge_index(pairs)
    tt <- graph_tensors(g, dev)
    z <- model_forward(model, tt)
    if (type == "embedding") return(as_r_array(z))
    ea <- if (!is.null(tt$edge_attr) && ncol(pairs) == ncol(g$edge_index)) tt$edge_attr
    out <- object$decoder(z, torch_tensor(pairs, dtype = torch_long(), device = dev), ea)
    format_predictions(out, type, object, seq_len(ncol(pairs)), "edge")
  })
}

# Graph task embeddings are one row per graph: node embeddings are pooled
# with the model's own readout (`config$pool`, mean when it has none).
pool_graph_embedding <- function(model, h, bt) {
  ng <- bt$num_graphs %||% 1L
  if (h$size(1L) == ng) return(h)
  pool <- tryCatch(model$config$pool, error = function(e) NULL) %||% "mean"
  switch(pool,
    sum = global_add_pool(h, bt$batch, ng),
    max = global_max_pool(h, bt$batch, ng),
    global_mean_pool(h, bt$batch, ng))
}

format_predictions <- function(out, type, object, ids, id_name) {
  classif <- grepl("classification", object$task)
  if (type == "response" || type == "embedding") {
    return(as_r_array(out))
  }
  if (classif) {
    prob <- as_r_array(nnf_softmax(out, dim = 2L))
    lv <- object$levels %||% as.character(seq_len(ncol(prob)))
    colnames(prob) <- lv
    cls <- factor(lv[max.col(prob, ties.method = "first")], levels = lv)
    if (type == "prob") return(prob)
    if (type == "class") return(cls)
    df <- data.frame(.row = seq_along(cls), id = ids, .pred_class = cls)
    names(df)[2L] <- id_name
    pr <- as.data.frame(prob)
    names(pr) <- paste0(".prob_", lv)
    return(structure(cbind(df, pr), class = c("graph_predictions", "data.frame")))
  }
  val <- as_r_array(out)
  if (is.matrix(val) && ncol(val) == 1L) val <- val[, 1L]
  if (type %in% c("class", "prob")) stopf("`type = \"%s\"` needs a classification task.", type)
  df <- data.frame(.row = seq_len(NROW(val)), id = ids)
  names(df)[2L] <- id_name
  if (is.matrix(val)) {
    pv <- as.data.frame(val)
    names(pv) <- paste0(".pred_", seq_len(ncol(val)))
    df <- cbind(df, pv)
  } else {
    df$.pred <- val
  }
  structure(df, class = c("graph_predictions", "data.frame"))
}

#' Evaluate a trained graph model
#'
#' Computes task metrics on a held out set, using [evaluate_predictions()].
#'
#' * Node tasks: nodes in `new_data`'s `split` mask (`"test"` by default).
#' * Graph tasks: every graph in `new_data`.
#' * Link prediction: `split$test_pos` against `split$test_neg` (or the
#'   validation pairs with `split = "val"`), with message passing on the
#'   training graph only.
#' * Edge tasks: edges in the `split` edge mask.
#'
#' @param fit A `graph_fit`.
#' @param new_data Evaluation data; defaults to the training data object.
#' @param split `"test"`, `"val"` or `"train"`.
#' @param ks Cut-offs for Hits@K.
#' @return A data frame with `metric` and `value`.
#' @export
evaluate_graph_model <- function(fit, new_data = NULL, split = c("test", "val", "train"),
                                 ks = c(1L, 10L, 50L)) {
  split <- match.arg(split)
  task <- fit$task
  classif <- grepl("classification", task)
  data <- new_data %||% fit$data
  if (task %in% c("node_classification", "node_regression")) {
    if (inherits(data, "hetero_graph_data")) {
      nd <- data$nodes[[fit$target_type]]
      mask <- nd[[paste0(split, "_mask")]]
      y <- nd$y
    } else {
      mask <- data[[paste0(split, "_mask")]]
      y <- data$y
    }
    if (is.null(mask) || !any(mask)) stopf("No nodes in the %s mask.", split)
    p <- predict(fit, data, type = if (classif) "prob" else "response")
    return(eval_rows(fit, y, p, mask, classif))
  }
  if (task %in% c("graph_classification", "graph_regression")) {
    if (inherits(data, "graph_data")) data <- list(data)
    y <- stack_graph_labels(lapply(seq_len(length(data)), function(i) data[[i]]$graph_y))
    p <- predict(fit, data, type = if (classif) "prob" else "response")
    return(eval_rows(fit, y, p, rep(TRUE, NROW(p)), classif))
  }
  if (task == "link_prediction") {
    sp <- data
    pos <- sp[[paste0(split, "_pos")]]
    neg <- sp[[paste0(split, "_neg")]]
    if (split == "train") stopf("Use split = \"val\" or \"test\" for link prediction.")
    if (inherits(sp, "hetero_link_split")) {
      s_pos <- predict(fit, pairs = pos, type = "score")
      s_neg <- predict(fit, pairs = neg, type = "score")
    } else {
      s_pos <- predict(fit, sp$train_graph, type = "score", pairs = pos)
      s_neg <- predict(fit, sp$train_graph, type = "score", pairs = neg)
    }
    return(evaluate_predictions(s_pos, s_pos, "link", neg = s_neg, ks = ks))
  }
  mask <- data[[paste0("edge_", split, "_mask")]]
  if (is.null(mask) || !any(mask)) stopf("No edges in the %s edge mask.", split)
  p <- predict(fit, data, type = if (classif) "prob" else "response")
  eval_rows(fit, data$edge_y, p, mask, classif)
}

# Metrics on the rows selected by `mask`. Classification truth is coded with
# the fitted class levels; regression keeps every target column, and the
# truth and prediction matrices must have the same shape.
eval_rows <- function(fit, y, p, mask, classif) {
  truth <- take_rows(y, mask)
  if (classif) {
    truth <- prep_labels(truth, TRUE, fit$levels %||% colnames(p))
    return(evaluate_predictions(truth, p[mask, , drop = FALSE], "classification"))
  }
  pred <- as.matrix(p)[mask, , drop = FALSE]
  truth <- as.matrix(truth)
  if (!identical(dim(truth), dim(pred))) {
    stopf("Regression targets have %d columns but the model predicts %d.", ncol(truth), ncol(pred))
  }
  evaluate_predictions(as.vector(truth), as.vector(pred), "regression")
}
