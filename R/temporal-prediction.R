#' Predict with a temporal model
#'
#' Continues a fitted temporal model on new inputs. Every method uses only
#' information available at the prediction time.
#'
#' * **TGN** ([train_tgn()]): scores candidate pairs `pairs` at times
#'   `time`. The fit keeps the memory, last update times and neighbor store
#'   after all training, validation and test events. Events observed since
#'   then can be given as `new_data` (a `temporal_graph_data` with times at
#'   or after the last event of the fit); they update the state first, in
#'   the same batches as during training, and no score uses an event at or
#'   after its own time. Prediction times must not precede the last
#'   observed event.
#' * **Dynamic link prediction** ([train_dynamic_link_model()]): scores
#'   `pairs` as links of snapshot `step` of `new_data` (a
#'   `snapshot_sequence`), from the recurrent state built on snapshots
#'   `1, ..., step - 1` only.
#' * **Spatiotemporal forecasting** ([train_spatiotemporal_model()]):
#'   forecasts the targets `horizon` steps after each forecast origin in
#'   `ends`, from the `lags` feature steps ending at that origin of
#'   `new_data` (a `spatiotemporal_signal`).
#'
#' @param object A `temporal_fit`.
#' @param new_data New events (TGN), a `snapshot_sequence` (dynamic links)
#'   or a `spatiotemporal_signal` (forecasting).
#' @param pairs 2 by K matrix of node pairs to score (link models).
#' @param time Prediction time, one value or one per pair (TGN).
#' @param step Target snapshot index (dynamic links).
#' @param ends Forecast origins, time step indices (forecasting).
#' @param type `"prob"` or `"score"` (logit) for link models.
#' @param ... Unused.
#' @return Link models: a numeric vector with one entry per pair.
#'   Forecasting: a matrix with one row per origin and one column per node.
#' @examples
#' if (torch::torch_is_installed()) {
#'   sig <- dataset_synthetic_signal(seed = 1)
#'   fit <- train_spatiotemporal_model(sig, lags = 3, hidden_dim = 8, epochs = 2, seed = 1)
#'   dim(predict(fit, sig, ends = 10:12))
#' }
#' @export
predict.temporal_fit <- function(object, new_data = NULL, pairs = NULL, time = NULL, step = NULL,
                                 ends = NULL, type = c("prob", "score"), ...) {
  type <- match.arg(type)
  task <- object$task %||% "spatiotemporal forecasting"
  link_out <- function(s) if (type == "prob") stats::plogis(s) else s
  if (task == "continuous time link prediction") {
    model <- object$model
    model$eval()
    if (is.null(pairs) || is.null(time)) stopf("TGN predictions need `pairs` and `time`.")
    pairs <- normalize_edge_index(pairs)
    k <- ncol(pairs)
    time <- as.numeric(time)
    if (length(time) == 1L) time <- rep(time, k)
    if (length(time) != k) stopf("`time` needs one value or one per pair.")
    if (any(pairs < 1L | pairs > model$num_nodes)) stopf("`pairs` refer to nodes outside the model.")
    state <- object$state
    last <- object$last_time
    if (!is.null(new_data)) {
      if (!inherits(new_data, "temporal_graph_data")) stopf("`new_data` must be a temporal_graph_data.")
      if (length(new_data$t) && min(new_data$t) < last) {
        stopf("`new_data` events must not precede the last event of the fit (time %s).", format(last))
      }
      if (model$edge_dim > 0L && length(new_data$t) && is.null(new_data$edge_attr)) {
        stopf("The model expects event features.")
      }
      state <- tgn_absorb(model, state, new_data, object$batch_size %||% 200L)
      if (length(new_data$t)) last <- max(new_data$t)
    }
    if (any(time < last)) stopf("Prediction times must not precede the last observed event (time %s).", format(last))
    s <- with_no_grad({
      cur <- if (is.null(state$pending)) list(mem = state$mem, last_t = state$last_t)
             else model$update_memory(state$mem, state$last_t, state$pending)
      q <- c(pairs[1L, ], pairs[2L, ])
      nb <- if (model$embedding != "time") tgn_store_query(state$store, q)
      z <- model$embed(cur$mem, cur$last_t, q, c(time, time), nb)
      model$score(z[seq_len(k), ], z[k + seq_len(k), ])
    })
    return(link_out(as.vector(as_r_array(s))))
  }
  if (task == "dynamic link prediction") {
    snaps <- new_data
    if (!inherits(snaps, "snapshot_sequence")) stopf("`new_data` must be a snapshot_sequence.")
    if (is.null(pairs)) stopf("Dynamic link predictions need `pairs`.")
    K <- length(snaps$graphs)
    if (is.null(step) || step < 2L || step > K + 1L) stopf("`step` must be between 2 and %d.", K + 1L)
    if (snaps$num_nodes != object$num_nodes) stopf("`new_data` has a different number of nodes.")
    dev <- object$device
    cell <- object$model
    cell$eval()
    x0 <- if (object$identity_features) diag(snaps$num_nodes) else NULL
    p <- torch_tensor(normalize_edge_index(pairs), dtype = torch_long(), device = dev)
    s <- with_no_grad({
      h <- NULL
      for (k in seq_len(step - 1L)) {
        tt <- graph_tensors(snaps$graphs[[k]], dev)
        x <- if (!is.null(x0)) torch_tensor(x0, dtype = torch_float(), device = dev) else tt$x
        h <- cell(x, tt$edge_index, h, tt$edge_weight)
      }
      (h$index_select(1L, p[1L, ]) * h$index_select(1L, p[2L, ]))$sum(dim = 2L)
    })
    return(link_out(as.vector(as_r_array(s))))
  }
  sig <- new_data
  if (!inherits(sig, "spatiotemporal_signal")) stopf("`new_data` must be a spatiotemporal_signal.")
  steps <- dim(sig$features)[1L]
  if (is.null(ends)) ends <- steps
  ends <- as.integer(ends)
  if (any(ends < object$lags | ends > steps)) {
    stopf("Forecast origins must be between %d and %d.", object$lags, steps)
  }
  dev <- object$device
  model <- object$model
  model$eval()
  tt <- graph_tensors(sig$graph, dev)
  n <- sig$graph$num_nodes
  feat <- function(s) torch_tensor(matrix(sig$features[s, , ], nrow = n), dtype = torch_float(), device = dev)
  out <- with_no_grad(t(vapply(ends, function(e) {
    xs <- lapply(seq.int(e - object$lags + 1L, e), feat)
    as.vector(as_r_array(model(xs, tt$edge_index, tt$edge_weight)$outputs[[object$lags]]$squeeze(2L)))
  }, numeric(n))))
  dimnames(out) <- list(paste0("origin_", ends), NULL)
  out
}
