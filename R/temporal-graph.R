#' Temporal graph representations
#'
#' graphlearnR keeps three kinds of time varying graphs apart, because their
#' semantics differ:
#'
#' 1. **Continuous time event graphs** (`temporal_graph_data()`): a stream
#'    of timestamped interactions `(src, dst, t)`, possibly with event
#'    features and labels. Events are stored sorted by time (stable, so ties
#'    keep input order); `event_id` gives each event's position in the input.
#' 2. **Snapshot sequences** ([snapshot_sequence()], [temporal_snapshots()]):
#'    an ordered list of static graphs over the same node set, each with an
#'    interval `[start, end)`.
#' 3. **Spatiotemporal signals** ([spatiotemporal_signal()]): a fixed graph
#'    with node features that change over time steps.
#'
#' Conversions are explicit: [temporal_snapshots()] turns events into
#' snapshots with a stated interval rule; nothing converts between the kinds
#' implicitly.
#'
#' @param src,dst Integer vectors of one based node indices.
#' @param t Numeric (or `POSIXct`, `Date`) event times.
#' @param num_nodes Number of nodes; inferred when `NULL`.
#' @param x Optional static node features.
#' @param edge_attr Optional event features (one row per event).
#' @param y Optional event labels.
#' @param node_ids External node identifiers.
#' @param directed Whether interactions are directed.
#' @param metadata Named list.
#' @return An object of class `temporal_graph_data`.
#' @examples
#' tg <- temporal_graph_data(src = c(1, 2, 3, 1), dst = c(2, 3, 1, 3),
#'                           t = c(1, 2, 2, 5))
#' tg
#' snaps <- temporal_snapshots(tg, breaks = c(0, 2, 4, 6))
#' length(snaps$graphs)
#' @export
temporal_graph_data <- function(src, dst, t, num_nodes = NULL, x = NULL, edge_attr = NULL,
                                y = NULL, node_ids = NULL, directed = TRUE,
                                metadata = list()) {
  check_index(src, "src")
  check_index(dst, "dst")
  src <- as.integer(src)
  dst <- as.integer(dst)
  if (length(src) != length(dst) || length(src) != length(t)) {
    stopf("`src`, `dst` and `t` must have the same length.")
  }
  tn <- as.numeric(t)
  if (anyNA(tn) || any(!is.finite(tn))) stopf("Event times must be finite and not missing.")
  x <- as_feature_matrix(x, "x")
  edge_attr <- as_feature_matrix(edge_attr, "edge_attr")
  num_nodes <- num_nodes %||% max(c(n_rows(x), length(node_ids), src, dst, 0L), na.rm = TRUE)
  ord <- order(tn, method = "radix")
  tg <- structure(list(
    src = src[ord], dst = dst[ord], t = tn[ord], time_class = class(t),
    edge_attr = take_rows(edge_attr, ord), y = take_rows(y, ord), x = x,
    event_id = ord, num_nodes = as.integer(num_nodes),
    node_ids = node_ids %||% seq_len(num_nodes), directed = directed, metadata = metadata
  ), class = "temporal_graph_data")
  validate_temporal_graph(tg)
  tg
}

#' @rdname temporal_graph_data
#' @param tg A `temporal_graph_data` object.
#' @export
validate_temporal_graph <- function(tg) {
  n <- tg$num_nodes
  m <- length(tg$t)
  problems <- character()
  if (any(c(tg$src, tg$dst) < 1L | c(tg$src, tg$dst) > n)) {
    problems <- c(problems, sprintf("Event endpoints must be in 1:%d.", n))
  }
  if (is.unsorted(tg$t)) problems <- c(problems, "Events are not in time order.")
  if (!is.null(tg$edge_attr) && nrow(tg$edge_attr) != m) {
    problems <- c(problems, "`edge_attr` must have one row per event.")
  }
  if (!is.null(tg$y) && n_rows(tg$y) != m) problems <- c(problems, "`y` must have one entry per event.")
  if (!is.null(tg$x) && nrow(tg$x) != n) problems <- c(problems, "`x` must have one row per node.")
  if (length(problems)) {
    stop(paste(c("Invalid temporal_graph_data:", paste0("* ", problems)), collapse = "\n"),
         call. = FALSE)
  }
  invisible(tg)
}

#' @export
num_nodes.temporal_graph_data <- function(g) g$num_nodes

#' @export
num_edges.temporal_graph_data <- function(g, unique_pairs = FALSE) length(g$t)

#' @export
print.temporal_graph_data <- function(x, ...) {
  cat(sprintf("<temporal_graph_data> %d nodes, %d events, t in [%s, %s]\n", x$num_nodes,
              length(x$t), format(min(x$t, Inf)), format(max(x$t, -Inf))))
  invisible(x)
}

#' Select events in a time window
#'
#' @param tg A `temporal_graph_data` object.
#' @param start,end Window `[start, end)`; `NULL` means unbounded.
#' @return A `temporal_graph_data` with the selected events.
#' @examples
#' tg <- dataset_synthetic_temporal(seed = 1)
#' temporal_window(tg, start = 10, end = 20)
#' @export
temporal_window <- function(tg, start = NULL, end = NULL) {
  keep <- rep(TRUE, length(tg$t))
  if (!is.null(start)) keep <- keep & tg$t >= as.numeric(start)
  if (!is.null(end)) keep <- keep & tg$t < as.numeric(end)
  out <- tg
  for (f in c("src", "dst", "t", "event_id")) out[[f]] <- tg[[f]][keep]
  out$edge_attr <- take_rows(tg$edge_attr, which(keep))
  out$y <- take_rows(tg$y, which(keep))
  out
}

#' Temporal train, validation and test split
#'
#' Splits events by time: training events have `t < val_time`, validation
#' events `val_time <= t < test_time`, test events `t >= test_time`. Models
#' evaluated at time `t` may use any event strictly before `t`, including
#' earlier validation events, but never later ones. With proportions instead
#' of times, the cut points are event time quantiles.
#'
#' @param tg A `temporal_graph_data` object.
#' @param val_time,test_time Cut points.
#' @param val,test Proportions used when the cut points are `NULL`.
#' @return A list with `train`, `val`, `test` (each a `temporal_graph_data`)
#'   and the cut points.
#' @examples
#' tg <- dataset_synthetic_temporal(seed = 1)
#' sp <- split_temporal(tg)
#' lengths(sp[c("train", "val", "test")])
#' @export
split_temporal <- function(tg, val_time = NULL, test_time = NULL, val = 0.15, test = 0.15) {
  if (is.null(val_time) || is.null(test_time)) {
    q <- stats::quantile(tg$t, c(1 - val - test, 1 - test), names = FALSE, type = 1)
    val_time <- val_time %||% q[1L]
    test_time <- test_time %||% q[2L]
  }
  if (test_time < val_time) stopf("`test_time` must not precede `val_time`.")
  list(train = temporal_window(tg, end = val_time),
       val = temporal_window(tg, val_time, test_time),
       test = temporal_window(tg, start = test_time),
       val_time = val_time, test_time = test_time)
}

#' Most recent temporal neighbors
#'
#' For each query `(node, time)`, returns up to `k` most recent events that
#' involve `node` and happened **strictly before** `time` (or at or before
#' `time` with `strict = FALSE`). This is the neighborhood a continuous time
#' model such as TGN or TGAT may use at prediction time; no event at or after
#' the query time is ever returned with the default `strict = TRUE`.
#'
#' For directed graphs only events with `dst == node` (incoming messages) are
#' used unless `direction = "both"`.
#'
#' @param tg A `temporal_graph_data` object.
#' @param nodes Query nodes.
#' @param times Query times (recycled).
#' @param k Neighbors per query.
#' @param strict Exclude events at exactly the query time.
#' @param direction `"in"` or `"both"`.
#' @return A data frame with `query`, `node`, `neighbor`, `t` and `event`
#'   (position in the sorted event list).
#' @examples
#' tg <- dataset_synthetic_temporal(seed = 1)
#' nb <- temporal_neighbors(tg, nodes = tg$dst[50], times = tg$t[50], k = 5)
#' nb
#' @export
temporal_neighbors <- function(tg, nodes, times, k = 10L, strict = TRUE,
                               direction = c("in", "both")) {
  direction <- match.arg(direction)
  times <- rep_len(as.numeric(times), length(nodes))
  both <- direction == "both" || !tg$directed
  ev <- data.frame(node = tg$dst, nb = tg$src, t = tg$t, e = seq_along(tg$t))
  if (both) ev <- rbind(ev, data.frame(node = tg$src, nb = tg$dst, t = tg$t, e = seq_along(tg$t)))
  by_node <- split(seq_len(nrow(ev)), factor(ev$node, levels = seq_len(tg$num_nodes)))
  res <- lapply(seq_along(nodes), function(q) {
    idx <- by_node[[nodes[q]]]
    if (!length(idx)) return(NULL)
    ok <- if (strict) ev$t[idx] < times[q] else ev$t[idx] <= times[q]
    idx <- idx[ok]
    if (!length(idx)) return(NULL)
    idx <- idx[order(-ev$t[idx], -ev$e[idx])][seq_len(min(k, length(idx)))]
    data.frame(query = q, node = nodes[q], neighbor = ev$nb[idx], t = ev$t[idx], event = ev$e[idx])
  })
  out <- do.call(rbind, res)
  if (is.null(out)) {
    out <- data.frame(query = integer(), node = integer(), neighbor = integer(),
                      t = numeric(), event = integer())
  }
  out
}

#' Snapshot sequences
#'
#' `snapshot_sequence()` wraps an ordered list of graphs on the same nodes.
#' `temporal_snapshots()` builds one from events: snapshot `k` contains the
#' events with `breaks[k] <= t < breaks[k + 1]` (the last interval is closed
#' on the right), or with `cumulative = TRUE` all events with
#' `t < breaks[k + 1]`. Repeated interactions become edge weights (counts).
#'
#' @param graphs List of `graph_data` objects with equal `num_nodes`.
#' @param start,end Interval bounds per snapshot.
#' @return An object of class `snapshot_sequence`.
#' @examples
#' tg <- dataset_synthetic_temporal(seed = 1)
#' snaps <- temporal_snapshots(tg, breaks = 5)
#' length(snaps$graphs)
#' @export
snapshot_sequence <- function(graphs, start = seq_along(graphs) - 1, end = seq_along(graphs)) {
  n <- vapply(graphs, num_nodes, integer(1))
  if (length(unique(n)) > 1L) stopf("All snapshots must have the same number of nodes.")
  if (is.unsorted(start)) stopf("Snapshots must be in time order.")
  structure(list(graphs = graphs, start = as.numeric(start), end = as.numeric(end),
                 num_nodes = n[1L]), class = "snapshot_sequence")
}

#' @rdname snapshot_sequence
#' @param tg A `temporal_graph_data` object.
#' @param breaks Number of equal width intervals, or a sorted vector of cut
#'   points.
#' @param cumulative Accumulate all earlier events into each snapshot.
#' @export
temporal_snapshots <- function(tg, breaks = 10L, cumulative = FALSE) {
  if (length(breaks) == 1L) {
    breaks <- seq(min(tg$t), max(tg$t), length.out = breaks + 1L)
  }
  breaks <- as.numeric(breaks)
  k <- length(breaks) - 1L
  graphs <- lapply(seq_len(k), function(i) {
    last <- i == k
    hi <- if (last) tg$t <= breaks[i + 1L] else tg$t < breaks[i + 1L]
    keep <- if (cumulative) hi else hi & tg$t >= breaks[i]
    s <- tg$src[keep]
    d <- tg$dst[keep]
    if (!tg$directed) {
      lo <- pmin(s, d)
      d <- pmax(s, d)
      s <- lo
    }
    key <- paste(s, d)
    u <- !duplicated(key)
    w <- as.vector(table(factor(key, levels = unique(key))))
    graph_data(x = tg$x, edge_index = rbind(s[u], d[u]),
               edge_weight = if (any(keep)) w, num_nodes = tg$num_nodes,
               directed = tg$directed, node_ids = tg$node_ids, validate = FALSE)
  })
  snapshot_sequence(graphs, breaks[-length(breaks)], breaks[-1L])
}

#' @export
length.snapshot_sequence <- function(x) length(x$graphs)

#' @export
print.snapshot_sequence <- function(x, ...) {
  cat(sprintf("<snapshot_sequence> %d snapshots of %d nodes\n", length(x$graphs), x$num_nodes))
  invisible(x)
}

#' Spatiotemporal signals on a fixed graph
#'
#' A graph whose structure is fixed while node features change over time,
#' such as traffic sensors on a road network.
#'
#' @param graph A `graph_data` object (its `x` is ignored).
#' @param features Array `T` by `N` by `F` (or a `T` by `N` matrix for one
#'   feature).
#' @param targets Optional `T` by `N` matrix of targets; default the first
#'   feature.
#' @param times Optional time stamps.
#' @return An object of class `spatiotemporal_signal`.
#' @examples
#' g <- dataset_karate()
#' sig <- spatiotemporal_signal(g, matrix(rnorm(10 * 34), 10, 34))
#' sig
#' @export
spatiotemporal_signal <- function(graph, features, targets = NULL, times = NULL) {
  if (is.matrix(features)) features <- array(features, c(dim(features), 1L))
  if (length(dim(features)) != 3L || dim(features)[2L] != graph$num_nodes) {
    stopf("`features` must be a T by N by F array with N = %d.", graph$num_nodes)
  }
  targets <- targets %||% features[, , 1L]
  if (!is.matrix(targets)) targets <- matrix(targets, nrow = dim(features)[1L])
  structure(list(graph = graph, features = features, targets = targets,
                 times = times %||% seq_len(dim(features)[1L])),
            class = "spatiotemporal_signal")
}

#' @export
print.spatiotemporal_signal <- function(x, ...) {
  d <- dim(x$features)
  cat(sprintf("<spatiotemporal_signal> %d steps, %d nodes, %d features\n", d[1L], d[2L], d[3L]))
  invisible(x)
}

#' Recurrent graph convolution cell
#'
#' A GRU whose input and hidden transforms are graph convolutions:
#' \deqn{z = \sigma(G_{xz}(x) + G_{hz}(h)),\quad r = \sigma(G_{xr}(x) + G_{hr}(h))}
#' \deqn{\tilde h = \tanh(G_{xh}(x) + G_{hh}(r \odot h)),\quad h' = z \odot h + (1 - z) \odot \tilde h}
#' with \eqn{G} a [gcn_conv()]. This follows GConvGRU (Seo et al. 2018) with
#' the Chebyshev convolution replaced by a first order GCN.
#'
#' @param in_features Input feature size.
#' @param hidden_features Hidden state size.
#' @return An `nn_module` called as `cell(x, edge_index, h, edge_weight)`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   cell <- graph_gru_cell(3, 8)
#'   g <- dataset_karate()
#'   x <- torch::torch_randn(34, 3)
#'   ei <- graph_tensors(g)$edge_index
#'   cell(x, ei)$shape
#' }
#' @export
graph_gru_cell <- nn_module(
  "graph_gru_cell",
  initialize = function(in_features, hidden_features) {
    self$hidden <- hidden_features
    self$xz <- gcn_conv(in_features, hidden_features)
    self$hz <- gcn_conv(hidden_features, hidden_features)
    self$xr <- gcn_conv(in_features, hidden_features)
    self$hr <- gcn_conv(hidden_features, hidden_features)
    self$xh <- gcn_conv(in_features, hidden_features)
    self$hh <- gcn_conv(hidden_features, hidden_features)
  },
  forward = function(x, edge_index, h = NULL, edge_weight = NULL) {
    if (is.null(h)) h <- torch_zeros(x$size(1L), self$hidden, device = x$device)
    z <- torch_sigmoid(self$xz(x, edge_index, edge_weight) + self$hz(h, edge_index, edge_weight))
    r <- torch_sigmoid(self$xr(x, edge_index, edge_weight) + self$hr(h, edge_index, edge_weight))
    ht <- torch_tanh(self$xh(x, edge_index, edge_weight) + self$hh(r * h, edge_index, edge_weight))
    z * h + (1 - z) * ht
  }
)

#' Evolving GCN cell (EvolveGCN-O)
#'
#' The EvolveGCN-O cell of Pareja et al. (2020): the GCN weight matrix, not a
#' node state, is the recurrent state. At each snapshot
#' \deqn{W_t = \mathrm{GRU}(W_{t-1}, W_{t-1}),\quad
#'       H_t = \mathrm{ReLU}(\hat A_t X_t W_t),}
#' with the rows of \eqn{W} as the GRU batch and \eqn{\hat A_t} the GCN
#' propagation matrix of snapshot \eqn{t} (self loops, symmetric
#' normalization). Because only weights evolve, nodes that appear late are
#' handled like any other node.
#'
#' The call signature matches [graph_gru_cell()] so the cell can replace it
#' in [train_dynamic_link_model()]. The evolving weight is kept inside the
#' module between calls: a call with `h = NULL` restarts from the learned
#' initial weight \eqn{W_0}; any other `h` (the previous output) means
#' "continue", and its value is not used.
#'
#' @param in_features Input feature size.
#' @param hidden_features Output size.
#' @return An `nn_module` called as `cell(x, edge_index, h, edge_weight)`.
#' @references Pareja et al. (2020) EvolveGCN: evolving graph convolutional
#'   networks for dynamic graphs. AAAI.
#' @examples
#' if (torch::torch_is_installed()) {
#'   cell <- evolve_gcn_cell(3, 8)
#'   ei <- graph_tensors(dataset_karate())$edge_index
#'   x <- torch::torch_randn(34, 3)
#'   h1 <- cell(x, ei)
#'   h2 <- cell(x, ei, h1)
#'   h2$shape
#' }
#' @export
evolve_gcn_cell <- nn_module(
  "evolve_gcn_cell",
  initialize = function(in_features, hidden_features) {
    self$hidden <- hidden_features
    self$W0 <- glorot_param(in_features, hidden_features)
    self$gru_x <- nn_linear(hidden_features, 3L * hidden_features)
    self$gru_h <- nn_linear(hidden_features, 3L * hidden_features)
    self$state <- new.env(parent = emptyenv())
  },
  gru = function(x, h) {
    # Standard GRU cell (Cho et al. 2014), gates in the order reset, update, new.
    gx <- self$gru_x(x)$chunk(3L, dim = 2L)
    gh <- self$gru_h(h)$chunk(3L, dim = 2L)
    r <- torch_sigmoid(gx[[1L]] + gh[[1L]])
    z <- torch_sigmoid(gx[[2L]] + gh[[2L]])
    n <- torch_tanh(gx[[3L]] + r * gh[[3L]])
    (1 - z) * n + z * h
  },
  forward = function(x, edge_index, h = NULL, edge_weight = NULL) {
    # ponytail: state kept in an environment, one sequence at a time; pass
    # W explicitly if several sequences must interleave.
    prev <- if (is.null(h) || is.null(self$state$W)) self$W0 else self$state$W
    W <- self$gru(prev, prev)
    self$state$W <- W
    nrm <- gcn_norm(edge_index, x$size(1L), edge_weight, dtype = x$dtype)
    nnf_relu(message_passing(x$matmul(W), nrm$edge_index, edge_weight = nrm$edge_weight,
                             num_nodes = x$size(1L)))
  }
)

#' Recurrent temporal GNN
#'
#' A [graph_gru_cell()] followed by a linear readout per node and step.
#' `forward(x_seq, edge_index_seq, edge_weight_seq)` takes lists of per step
#' tensors (one graph per step, or the same graph repeated) and returns a list
#' with per step outputs and the final hidden state. Outputs at step `s`
#' depend only on inputs at steps `1, ..., s`.
#'
#' @param input_dim Node feature size.
#' @param hidden_dim Hidden size.
#' @param output_dim Output size (per node), or `NULL` to return hidden
#'   states.
#' @return An `nn_module` of class `temporal_gnn`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   m <- temporal_gnn_model(input_dim = 2, hidden_dim = 8, output_dim = 1)
#'   m
#' }
#' @export
temporal_gnn_model <- nn_module(
  "temporal_gnn",
  initialize = function(input_dim, hidden_dim = 32L, output_dim = 1L) {
    self$cell <- graph_gru_cell(input_dim, hidden_dim)
    if (!is.null(output_dim)) self$head <- nn_linear(hidden_dim, output_dim)
  },
  forward = function(x_seq, edge_index_seq, edge_weight_seq = NULL, h = NULL) {
    steps <- length(x_seq)
    outs <- vector("list", steps)
    for (s in seq_len(steps)) {
      ei <- if (is.list(edge_index_seq)) edge_index_seq[[s]] else edge_index_seq
      w <- if (is.list(edge_weight_seq)) edge_weight_seq[[s]] else edge_weight_seq
      h <- self$cell(x_seq[[s]], ei, h, w)
      outs[[s]] <- if (is.null(self$head)) h else self$head(h)
    }
    list(outputs = outs, hidden = h)
  }
)

#' Train a spatiotemporal forecaster
#'
#' Fits a [temporal_gnn_model()] to predict the target at step `s + horizon`
#' from features at steps `s - lags + 1, ..., s`. Windows are split in time
#' order: the first `train` fraction of target steps trains the model, the
#' next `val` fraction validates, and the rest is the test period. Windows at
#' the end of the training and validation periods whose target is observed
#' after the first forecast origin of the next period are dropped (a purge of
#' `horizon - 1` windows), so no training target is later than a validation
#' or test forecast origin.
#'
#' @param signal A [spatiotemporal_signal()].
#' @param lags Input window length.
#' @param horizon Steps ahead.
#' @param hidden_dim Hidden size.
#' @param epochs Epochs.
#' @param lr Learning rate.
#' @param train,val Fractions of windows for training and validation.
#' @param seed Optional integer seed.
#' @param device Device.
#' @return An object of class `temporal_fit` with the model, history,
#'   window split and test metrics (`test_metrics`), alongside a naive
#'   last value baseline (`baseline_metrics`).
#' @examples
#' if (torch::torch_is_installed()) {
#'   sig <- dataset_synthetic_signal(seed = 1)
#'   fit <- train_spatiotemporal_model(sig, lags = 3, hidden_dim = 8, epochs = 3, seed = 1)
#'   fit
#' }
#' @export
train_spatiotemporal_model <- function(signal, lags = 4L, horizon = 1L, hidden_dim = 32L,
                                       epochs = 50L, lr = 0.01, train = 0.7, val = 0.15,
                                       seed = NULL, device = NULL) {
  dev <- resolve_device(device)
  if (!is.null(seed)) torch_manual_seed(seed)
  steps <- dim(signal$features)[1L]
  ends <- seq.int(lags, steps - horizon)
  if (length(ends) < 3L) stopf("Too few time steps for lags = %d and horizon = %d.", lags, horizon)
  n_tr <- floor(train * length(ends))
  n_va <- floor(val * length(ends))
  sets <- list(train = ends[seq_len(n_tr)], val = ends[n_tr + seq_len(n_va)],
               test = ends[-seq_len(n_tr + n_va)])
  # Purge: an earlier set may not use a target observed after the forecast
  # origin of the next set, so drop its last `horizon - 1` windows.
  purge <- function(a, b) if (length(b)) a[a + horizon <= min(b)] else a
  sets$val <- purge(sets$val, sets$test)
  sets$train <- purge(sets$train, if (length(sets$val)) sets$val else sets$test)
  tt <- graph_tensors(signal$graph, dev)
  model <- temporal_gnn_model(dim(signal$features)[3L], hidden_dim, 1L)
  model$to(device = dev)
  opt <- optim_adam(model$parameters, lr = lr)
  feat <- function(s) torch_tensor(matrix(signal$features[s, , ], nrow = signal$graph$num_nodes),
                                   dtype = torch_float(), device = dev)
  run <- function(end) {
    xs <- lapply(seq.int(end - lags + 1L, end), feat)
    out <- model(xs, tt$edge_index, tt$edge_weight)
    out$outputs[[lags]]$squeeze(2L)
  }
  target <- function(end) torch_tensor(signal$targets[end + horizon, ], dtype = torch_float(),
                                       device = dev)
  eval_set <- function(set) {
    if (!length(set)) return(NA_real_)
    model$eval()
    with_no_grad(mean(vapply(set, function(e) as.numeric(nnf_mse_loss(run(e), target(e))$item()),
                             numeric(1))))
  }
  history <- data.frame(epoch = integer(), train_loss = numeric(), val_loss = numeric())
  for (ep in seq_len(epochs)) {
    model$train()
    order_tr <- with_optional_seed(if (is.null(seed)) NULL else seed + ep, sample(sets$train))
    losses <- vapply(order_tr, function(e) {
      opt$zero_grad()
      loss <- nnf_mse_loss(run(e), target(e))
      loss$backward()
      opt$step()
      as.numeric(loss$item())
    }, numeric(1))
    history[ep, ] <- list(ep, mean(losses), eval_set(sets$val))
  }
  model$eval()
  pred <- with_no_grad(lapply(sets$test, function(e) as_r_array(run(e))))
  truth <- lapply(sets$test, function(e) signal$targets[e + horizon, ])
  naive <- lapply(sets$test, function(e) signal$targets[e, ])
  mets <- function(p) c(mse = metric_mse(unlist(truth), unlist(p)),
                        mae = metric_mae(unlist(truth), unlist(p)))
  structure(list(model = model, history = history, windows = sets, lags = lags,
                 horizon = horizon, test_metrics = if (length(pred)) mets(pred),
                 baseline_metrics = if (length(pred)) mets(naive), device = dev,
                 seed = seed, task = "spatiotemporal forecasting"), class = "temporal_fit")
}

#' @export
print.temporal_fit <- function(x, ...) {
  cat(sprintf("<temporal_fit> %s, %d epochs\n", x$task %||% "spatiotemporal forecasting",
              nrow(x$history)))
  if (!is.null(x$test_metrics)) {
    cat("  test:     ", paste(names(x$test_metrics), signif(x$test_metrics, 4), sep = " = ", collapse = ", "), "\n")
    cat("  baseline: ", paste(names(x$baseline_metrics), signif(x$baseline_metrics, 4), sep = " = ", collapse = ", "), "\n")
  }
  invisible(x)
}

#' Dynamic link prediction on snapshots
#'
#' Trains a recurrent graph encoder on a [snapshot_sequence()]: the hidden
#' state after snapshot `k` (computed from snapshots `1, ..., k` only) scores
#' the links of snapshot `k + 1` against negatives drawn from non-edges of
#' snapshot `k + 1`. Transitions whose target snapshot index is in
#' `train_steps` are used for training; the remaining transitions are
#' evaluated, in time order, with the state carried forward through every
#' earlier snapshot. No snapshot after `k` ever influences a score for
#' snapshot `k + 1`.
#'
#' @param snapshots A `snapshot_sequence`.
#' @param train_steps Target snapshot indices used for training (default the
#'   first 70 percent of transitions).
#' @param hidden_dim Hidden size.
#' @param epochs Epochs.
#' @param lr Learning rate.
#' @param neg_ratio Negatives per positive.
#' @param seed Optional integer seed.
#' @param device Device.
#' @param cell `"gru"` for a [graph_gru_cell()] (node states evolve) or
#'   `"evolve"` for an [evolve_gcn_cell()] (GCN weights evolve, EvolveGCN-O).
#' @return An object of class `temporal_fit` with per step test metrics
#'   (`step_metrics`) and averages (`test_metrics`).
#' @examples
#' if (torch::torch_is_installed()) {
#'   snaps <- temporal_snapshots(dataset_synthetic_temporal(seed = 1), breaks = 8)
#'   fit <- train_dynamic_link_model(snaps, hidden_dim = 8, epochs = 3, seed = 1)
#'   fit
#' }
#' @export
train_dynamic_link_model <- function(snapshots, train_steps = NULL, hidden_dim = 32L,
                                     epochs = 50L, lr = 0.01, neg_ratio = 1, seed = NULL,
                                     device = NULL, cell = c("gru", "evolve")) {
  cell_type <- match.arg(cell)
  dev <- resolve_device(device)
  if (!is.null(seed)) torch_manual_seed(seed)
  K <- length(snapshots$graphs)
  if (K < 3L) stopf("Need at least three snapshots.")
  targets <- 2:K
  train_steps <- train_steps %||% targets[seq_len(max(1L, floor(0.7 * length(targets))))]
  eval_steps <- setdiff(targets, train_steps)
  if (any(eval_steps < max(train_steps))) {
    stopf("Evaluation snapshots must come after all training snapshots.")
  }
  n <- snapshots$num_nodes
  feats <- lapply(snapshots$graphs, function(g) g$x)
  x0 <- if (!is.null(feats[[1L]])) feats[[1L]] else diag(n)
  tts <- lapply(snapshots$graphs, graph_tensors, device = dev)
  xt <- lapply(seq_len(K), function(k) {
    if (!is.null(tts[[k]]$x)) tts[[k]]$x else torch_tensor(x0, dtype = torch_float(), device = dev)
  })
  cell <- if (cell_type == "gru") graph_gru_cell(ncol(x0), hidden_dim)
          else evolve_gcn_cell(ncol(x0), hidden_dim)
  cell$to(device = dev)
  opt <- optim_adam(cell$parameters, lr = lr)
  pairs <- function(k, s) {
    g <- snapshots$graphs[[k]]
    pos <- g$edge_index
    neg <- negative_sampling(pos, num_nodes = n, num_samples = max(1L, ceiling(neg_ratio * ncol(pos))),
                             directed = g$directed, seed = s)
    list(pos = torch_tensor(pos, dtype = torch_long(), device = dev),
         neg = torch_tensor(neg, dtype = torch_long(), device = dev))
  }
  dot <- function(h, p) (h$index_select(1L, p[1L, ]) * h$index_select(1L, p[2L, ]))$sum(dim = 2L)
  history <- data.frame(epoch = integer(), train_loss = numeric())
  for (ep in seq_len(epochs)) {
    ep_seeds <- sub_seeds(if (is.null(seed)) NULL else seed + ep, max(train_steps))
    cell$train()
    h <- NULL
    total <- 0
    opt$zero_grad()
    for (k in seq_len(max(train_steps) - 1L)) {
      h <- cell(xt[[k]], tts[[k]]$edge_index, h, tts[[k]]$edge_weight)
      if ((k + 1L) %in% train_steps && ncol(snapshots$graphs[[k + 1L]]$edge_index)) {
        pr <- pairs(k + 1L, ep_seeds[k])
        sp <- dot(h, pr$pos)
        sn <- dot(h, pr$neg)
        loss <- nnf_binary_cross_entropy_with_logits(
          torch_cat(list(sp, sn)), torch_cat(list(torch_ones_like(sp), torch_zeros_like(sn))))
        total <- total + loss
      }
    }
    if (!inherits(total, "torch_tensor")) stopf("No training transition has edges.")
    total$backward()
    opt$step()
    history[ep, ] <- list(ep, as.numeric(total$item()) / length(train_steps))
  }
  cell$eval()
  step_metrics <- with_no_grad({
    h <- NULL
    rows <- list()
    for (k in seq_len(K - 1L)) {
      h <- cell(xt[[k]], tts[[k]]$edge_index, h, tts[[k]]$edge_weight)
      if ((k + 1L) %in% eval_steps && ncol(snapshots$graphs[[k + 1L]]$edge_index)) {
        pr <- pairs(k + 1L, if (is.null(seed)) NULL else seed + 7919L * k)
        sp <- as.vector(as_r_array(dot(h, pr$pos)))
        sn <- as.vector(as_r_array(dot(h, pr$neg)))
        y <- c(rep(TRUE, length(sp)), rep(FALSE, length(sn)))
        rows[[length(rows) + 1L]] <- data.frame(
          step = k + 1L, roc_auc = metric_roc_auc(y, c(sp, sn)),
          average_precision = metric_average_precision(y, c(sp, sn)))
      }
    }
    do.call(rbind, rows)
  })
  structure(list(model = cell, history = history, train_steps = train_steps,
                 eval_steps = eval_steps, step_metrics = step_metrics,
                 test_metrics = if (!is.null(step_metrics)) colMeans(step_metrics[, -1L, drop = FALSE]),
                 baseline_metrics = if (length(eval_steps)) dynamic_link_baseline(snapshots, eval_steps, neg_ratio, seed),
                 task = "dynamic link prediction", device = dev, seed = seed,
                 num_nodes = n, identity_features = is.null(feats[[1L]])),
            class = "temporal_fit")
}

#' Heuristic baseline for dynamic link prediction
#'
#' Scores the links of snapshot `k` with a heuristic computed from snapshots
#' `1, ..., k - 1` only: the number of earlier interactions between the pair
#' (`"history"`) or the product of earlier degrees (`"preferential"`), with
#' the same negative sampling as [train_dynamic_link_model()].
#'
#' @param snapshots A `snapshot_sequence`.
#' @param steps Target snapshot indices.
#' @param neg_ratio Negatives per positive.
#' @param seed Optional integer seed.
#' @param method `"history"` or `"preferential"`.
#' @return Named vector of mean ROC-AUC and average precision.
#' @examples
#' snaps <- temporal_snapshots(dataset_synthetic_temporal(seed = 1), breaks = 8)
#' dynamic_link_baseline(snaps, steps = 7:8, seed = 1)
#' @export
dynamic_link_baseline <- function(snapshots, steps, neg_ratio = 1, seed = NULL,
                                  method = c("history", "preferential")) {
  method <- match.arg(method)
  n <- snapshots$num_nodes
  res <- t(vapply(steps, function(k) {
    past <- snapshots$graphs[seq_len(k - 1L)]
    A <- Reduce(`+`, lapply(past, function(g) adjacency_from_graph(g, weighted = FALSE)))
    g <- snapshots$graphs[[k]]
    pos <- g$edge_index
    neg <- negative_sampling(pos, num_nodes = n, num_samples = max(1L, ceiling(neg_ratio * ncol(pos))),
                             directed = g$directed, seed = if (is.null(seed)) NULL else seed + 7919L * (k - 1L))
    sc <- function(p) {
      if (method == "history") as.numeric(A[cbind(p[1L, ], p[2L, ])])
      else Matrix::rowSums(A)[p[1L, ]] * Matrix::colSums(A)[p[2L, ]]
    }
    s <- c(sc(pos), sc(neg))
    y <- c(rep(TRUE, ncol(pos)), rep(FALSE, ncol(neg)))
    c(roc_auc = metric_roc_auc(y, s), average_precision = metric_average_precision(y, s))
  }, numeric(2)))
  colMeans(res)
}

#' Synthetic temporal data
#'
#' `dataset_synthetic_temporal()` simulates an interaction stream on a
#' community graph where pairs that interacted before tend to interact again,
#' so the past predicts the future. `dataset_synthetic_signal()` simulates
#' a diffusion process on a ring graph: each node's value moves towards the
#' mean of its neighbors plus noise.
#'
#' @param num_nodes Nodes.
#' @param num_events Events.
#' @param num_steps Time steps.
#' @param seed Optional integer seed.
#' @return A `temporal_graph_data` or a `spatiotemporal_signal`.
#' @examples
#' tg <- dataset_synthetic_temporal(seed = 1)
#' tg
#' sig <- dataset_synthetic_signal(seed = 1)
#' dim(sig$features)
#' @export
dataset_synthetic_temporal <- function(num_nodes = 40L, num_events = 800L, seed = NULL) {
  with_optional_seed(seed, {
    comm <- sample(rep_len(1:4, num_nodes))
    src <- dst <- integer(num_events)
    for (i in seq_len(num_events)) {
      if (i > 20L && stats::runif(1) < 0.6) {
        j <- sample.int(i - 1L, 1L)
        src[i] <- src[j]
        dst[i] <- dst[j]
      } else {
        src[i] <- sample.int(num_nodes, 1L)
        same <- setdiff(which(comm == comm[src[i]]), src[i])
        dst[i] <- if (stats::runif(1) < 0.8 && length(same)) same[sample.int(length(same), 1L)]
                  else sample(setdiff(seq_len(num_nodes), src[i]), 1L)
      }
    }
    t <- sort(stats::runif(num_events, 0, 100))
  })
  temporal_graph_data(src, dst, t, num_nodes = num_nodes, directed = FALSE,
                      metadata = list(name = "synthetic_temporal", seed = seed))
}

#' @rdname dataset_synthetic_temporal
#' @export
dataset_synthetic_signal <- function(num_nodes = 20L, num_steps = 60L, seed = NULL) {
  g <- graph_data(edge_index = rbind(seq_len(num_nodes), c(seq_len(num_nodes)[-1L], 1L)),
                  num_nodes = num_nodes, directed = FALSE)
  A <- as.matrix(adjacency_from_graph(g))
  P <- A / rowSums(A)
  with_optional_seed(seed, {
    v <- matrix(0, num_steps, num_nodes)
    v[1L, ] <- stats::rnorm(num_nodes)
    for (s in 2:num_steps) {
      v[s, ] <- 0.5 * v[s - 1L, ] + 0.45 * as.vector(P %*% v[s - 1L, ]) + 0.1 * stats::rnorm(num_nodes)
    }
  })
  spatiotemporal_signal(g, v)
}
