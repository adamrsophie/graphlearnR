#' Graph transformations
#'
#' Functions that return a modified copy of a `graph_data` object.
#'
#' * `add_self_loops()`: add a loop `(i, i)` to every node that has none.
#'   New loops get weight `fill_weight` and zero edge features.
#' * `remove_self_loops()`: drop all loops.
#' * `coalesce_edges()`: merge duplicate stored edges, summing weights and
#'   averaging edge features; keeps the first label.
#' * `induced_subgraph()`: keep `nodes` (in the given order) and the edges
#'   among them.
#' * `as_undirected()`: turn a directed graph into an undirected one, keeping
#'   one entry per unordered pair (attributes of the first occurrence).
#' * `normalize_features()`: standardize or row normalize node features.
#'   With `stats_mask`, means and scales come from those nodes only (for
#'   example the training nodes), so no information from held out nodes
#'   enters preprocessing.
#' * `add_degree_features()`: append in degree (optionally log scaled) or
#'   one-hot degree columns to `x`.
#' * `compose_transforms()`: chain transforms for [graph_dataset()].
#'
#' @param g A `graph_data` object.
#' @return A `graph_data` object.
#' @examples
#' g <- graph_data(x = diag(3), edge_index = rbind(c(1, 1, 2), c(2, 2, 3)))
#' has_duplicate_edges(g)
#' g2 <- coalesce_edges(g)
#' num_edges(g2)
#' num_edges(add_self_loops(g2))
#' @name graph-transforms
NULL

#' @rdname graph-transforms
#' @param fill_weight Weight of new self loops.
#' @export
add_self_loops <- function(g, fill_weight = 1) {
  ei <- g$edge_index
  has <- logical(g$num_nodes)
  has[ei[1L, ei[1L, ] == ei[2L, ]]] <- TRUE
  new <- which(!has)
  if (!length(new)) return(g)
  k <- length(new)
  g$edge_index <- cbind(ei, rbind(new, new, deparse.level = 0L))
  if (!is.null(g$edge_weight)) g$edge_weight <- c(g$edge_weight, rep(fill_weight, k))
  if (!is.null(g$edge_attr)) g$edge_attr <- rbind(g$edge_attr, matrix(0, k, ncol(g$edge_attr)))
  if (!is.null(g$edge_y)) g$edge_y <- c(g$edge_y, rep(NA, k))
  if (!is.null(g$edge_type)) g$edge_type <- c(g$edge_type, rep(NA, k))
  g$edge_pair <- c(g$edge_pair, max(g$edge_pair, 0L) + seq_len(k))
  for (f in c("edge_train_mask", "edge_val_mask", "edge_test_mask")) {
    if (!is.null(g[[f]])) g[[f]] <- c(g[[f]], rep(FALSE, k))
  }
  g
}

#' @rdname graph-transforms
#' @export
remove_self_loops <- function(g) {
  keep <- which(g$edge_index[1L, ] != g$edge_index[2L, ])
  edge_subset(g, keep)
}

#' @rdname graph-transforms
#' @export
coalesce_edges <- function(g) {
  ei <- g$edge_index
  if (!ncol(ei)) return(g)
  key <- paste(ei[1L, ], ei[2L, ])
  grp <- match(key, unique(key))
  first <- !duplicated(grp)
  out <- edge_subset(g, which(first))
  if (!is.null(g$edge_weight)) out$edge_weight <- as.vector(rowsum(g$edge_weight, grp, reorder = FALSE))
  if (!is.null(g$edge_attr)) {
    cnt <- tabulate(grp)
    out$edge_attr <- rowsum(g$edge_attr, grp, reorder = FALSE) / cnt
    dimnames(out$edge_attr) <- list(NULL, colnames(g$edge_attr))
  }
  out
}

#' @rdname graph-transforms
#' @param nodes Integer node indices to keep.
#' @export
induced_subgraph <- function(g, nodes) {
  nodes <- as.integer(nodes)
  if (anyDuplicated(nodes) || any(nodes < 1L | nodes > g$num_nodes)) {
    stopf("`nodes` must be unique indices in 1:%d.", g$num_nodes)
  }
  local <- integer(g$num_nodes)
  local[nodes] <- seq_along(nodes)
  keep <- which(local[g$edge_index[1L, ]] > 0L & local[g$edge_index[2L, ]] > 0L)
  out <- edge_subset(g, keep)
  out$edge_index <- rbind(local[g$edge_index[1L, keep]], local[g$edge_index[2L, keep]])
  out$num_nodes <- length(nodes)
  for (f in c("x", "y", "node_ids", "node_type", "train_mask", "val_mask", "test_mask")) {
    out[[f]] <- take_rows(g[[f]], nodes)
  }
  out
}

#' @rdname graph-transforms
#' @export
as_undirected <- function(g) {
  if (!g$directed) return(g)
  ei <- g$edge_index
  lo <- pmin(ei[1L, ], ei[2L, ])
  hi <- pmax(ei[1L, ], ei[2L, ])
  first <- which(!duplicated(paste(lo, hi)))
  graph_data(x = g$x, edge_index = rbind(ei[1L, first], ei[2L, first]), y = g$y,
             edge_attr = take_rows(g$edge_attr, first),
             edge_weight = take_rows(g$edge_weight, first),
             edge_y = take_rows(g$edge_y, first), graph_y = g$graph_y,
             graph_attr = g$graph_attr, num_nodes = g$num_nodes, directed = FALSE,
             node_ids = g$node_ids, node_type = g$node_type,
             edge_type = take_rows(g$edge_type, first), train_mask = g$train_mask,
             val_mask = g$val_mask, test_mask = g$test_mask, metadata = g$metadata)
}

#' @rdname graph-transforms
#' @param method `"standardize"` (zero mean, unit variance per column) or
#'   `"row"` (rows sum to one, as in Kipf and Welling 2017).
#' @param stats_mask Logical node mask used to compute column statistics.
#' @export
normalize_features <- function(g, method = c("standardize", "row"), stats_mask = NULL) {
  method <- match.arg(method)
  if (is.null(g$x)) stopf("The graph has no node features.")
  x <- g$x
  if (method == "row") {
    s <- rowSums(x)
    s[s == 0] <- 1
    g$x <- x / s
    return(g)
  }
  ref <- if (is.null(stats_mask)) x else x[stats_mask, , drop = FALSE]
  mu <- colMeans(ref)
  sdv <- apply(ref, 2L, stats::sd)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  g$x <- sweep(sweep(x, 2L, mu), 2L, sdv, "/")
  g$metadata$feature_center <- mu
  g$metadata$feature_scale <- sdv
  g
}

#' @rdname graph-transforms
#' @param one_hot Use one-hot degree columns capped at `max_degree`.
#' @param max_degree Cap for one-hot encoding.
#' @param log Use `log1p(degree)` for the numeric column.
#' @export
add_degree_features <- function(g, one_hot = FALSE, max_degree = 10L, log = TRUE) {
  d <- degree(g, "in")
  feat <- if (one_hot) {
    m <- matrix(0, g$num_nodes, max_degree + 1L)
    m[cbind(seq_len(g$num_nodes), pmin(d, max_degree) + 1L)] <- 1
    m
  } else {
    matrix(if (log) log1p(d) else as.double(d), ncol = 1L)
  }
  g$x <- if (is.null(g$x)) feat else cbind(g$x, feat)
  g
}

#' @rdname graph-transforms
#' @param ... Transform functions, applied left to right.
#' @export
compose_transforms <- function(...) {
  fs <- list(...)
  function(g) {
    for (f in fs) g <- f(g)
    g
  }
}
