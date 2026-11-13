#' Batch graphs into one disconnected graph
#'
#' Combines several graphs into a single graph whose components are the
#' input graphs. Node indices of graph `k` are shifted by the number of nodes
#' in graphs `1, ..., k - 1`, so no edge ever joins two input graphs. The
#' result records:
#'
#' * `batch`: graph membership of every node (`1:num_graphs`),
#' * `node_ptr`, `edge_ptr`: offsets, so the nodes of graph `k` are
#'   `(node_ptr[k] + 1):node_ptr[k + 1]`, and likewise for edges,
#' * `original_node_ids`: the `node_ids` of each node in its own graph,
#' * `graph_y` and `graph_attr` stacked one row (or element) per graph.
#'
#' A `graph_batch` is also a `graph_data`, so every layer and pooling
#' function accepts it. [unbatch()] reconstructs the input graphs.
#'
#' @param graphs A list of `graph_data` objects with the same number of node
#'   features, edge features and the same direction.
#' @return An object of class `c("graph_batch", "graph_data")`.
#' @examples
#' g1 <- graph_data(x = matrix(1, 2, 1), edge_index = rbind(1, 2), graph_y = 0)
#' g2 <- graph_data(x = matrix(2, 3, 1), edge_index = rbind(c(1, 2), c(2, 3)),
#'                  graph_y = 1)
#' b <- graph_batch(list(g1, g2))
#' b$num_graphs
#' b$edge_index
#' b$batch
#' length(unbatch(b))
#' @export
graph_batch <- function(graphs) {
  if (inherits(graphs, "graph_data")) {
    graphs <- list(graphs)
  }
  if (!is.list(graphs) || !length(graphs) ||
      !all(vapply(graphs, inherits, logical(1), "graph_data"))) {
    stopf("`graphs` must be a non-empty list of graph_data objects.")
  }
  if (any(vapply(graphs, inherits, logical(1), "graph_batch"))) {
    graphs <- unlist(lapply(graphs, function(g) {
      if (inherits(g, "graph_batch")) unbatch(g) else list(g)
    }), recursive = FALSE)
  }
  dirs <- vapply(graphs, function(g) g$directed, logical(1))
  if (length(unique(dirs)) > 1L) {
    stopf("Cannot batch directed and undirected graphs together.")
  }
  check_same <- function(f, what) {
    v <- vapply(graphs, f, integer(1))
    if (length(unique(v)) > 1L) {
      stopf("All graphs must have the same number of %s (found %s).", what,
            paste(unique(v), collapse = ", "))
    }
  }
  check_same(num_node_features, "node features")
  check_same(num_edge_features, "edge features")
  for (f in c("x", "y", "edge_weight", "edge_attr", "edge_y", "graph_y")) {
    has <- vapply(graphs, function(g) !is.null(g[[f]]), logical(1))
    if (any(has) && !all(has)) {
      stopf("`%s` is present in some graphs but not in others.", f)
    }
  }

  n <- vapply(graphs, num_nodes, integer(1))
  e <- vapply(graphs, function(g) ncol(g$edge_index), integer(1))
  p <- vapply(graphs, function(g) if (length(g$edge_pair)) max(g$edge_pair) else 0L, integer(1))
  node_ptr <- c(0L, cumsum(n))
  edge_ptr <- c(0L, cumsum(e))
  pair_ptr <- c(0L, cumsum(p))
  ei <- do.call(cbind, lapply(seq_along(graphs), function(k) {
    graphs[[k]]$edge_index + node_ptr[k]
  }))
  pick <- function(f) lapply(graphs, `[[`, f)
  graph_attr <- pick("graph_attr")
  graph_attr <- if (!all(vapply(graph_attr, is.null, logical(1)))) do.call(rbind, graph_attr)
  graph_y <- stack_graph_labels(pick("graph_y"))

  b <- new_graph_data(
    x = bind_rows_values(pick("x")),
    edge_index = if (length(ei)) ei else matrix(integer(), 2L, 0L),
    y = bind_rows_values(pick("y")),
    edge_attr = bind_rows_values(pick("edge_attr")),
    edge_weight = bind_rows_values(pick("edge_weight")),
    edge_y = bind_rows_values(pick("edge_y")),
    graph_y = graph_y,
    num_nodes = sum(n),
    directed = dirs[1L],
    node_ids = seq_len(sum(n)),
    node_type = bind_rows_values(pick("node_type")),
    edge_type = bind_rows_values(pick("edge_type")),
    edge_pair = unlist(lapply(seq_along(graphs), function(k) {
      graphs[[k]]$edge_pair + pair_ptr[k]
    }), use.names = FALSE) %||% integer(),
    train_mask = bind_rows_values(pick("train_mask")),
    val_mask = bind_rows_values(pick("val_mask")),
    test_mask = bind_rows_values(pick("test_mask")),
    metadata = list(graphs = pick("metadata")),
    class = "graph_batch"
  )
  b$graph_attr <- graph_attr
  b$num_graphs <- length(graphs)
  b$batch <- rep.int(seq_along(graphs), n)
  b$node_ptr <- node_ptr
  b$edge_ptr <- edge_ptr
  b$pair_ptr <- pair_ptr
  b$original_node_ids <- pick("node_ids")
  b
}

stack_graph_labels <- function(vals) {
  if (all(vapply(vals, is.null, logical(1)))) {
    return(NULL)
  }
  lens <- vapply(vals, length, integer(1))
  if (all(lens == 1L)) {
    return(bind_rows_values(vals))
  }
  if (length(unique(lens)) > 1L) {
    stopf("`graph_y` has different lengths across graphs.")
  }
  do.call(rbind, lapply(vals, as.double))
}

#' @rdname graph_batch
#' @param batch A `graph_batch` object.
#' @export
unbatch <- function(batch) {
  if (!inherits(batch, "graph_batch")) {
    stopf("`batch` must be a graph_batch object.")
  }
  lapply(seq_len(batch$num_graphs), function(k) {
    nodes <- seq.int(batch$node_ptr[k] + 1L, length.out = batch$node_ptr[k + 1L] - batch$node_ptr[k])
    edges <- seq.int(batch$edge_ptr[k] + 1L, length.out = batch$edge_ptr[k + 1L] - batch$edge_ptr[k])
    gy <- batch$graph_y
    gy <- if (is.null(gy)) NULL else if (is.matrix(gy)) gy[k, ] else gy[k]
    ga <- batch$graph_attr
    new_graph_data(
      x = take_rows(batch$x, nodes),
      edge_index = batch$edge_index[, edges, drop = FALSE] - batch$node_ptr[k],
      y = take_rows(batch$y, nodes),
      edge_attr = take_rows(batch$edge_attr, edges),
      edge_weight = take_rows(batch$edge_weight, edges),
      edge_y = take_rows(batch$edge_y, edges),
      graph_y = gy,
      graph_attr = if (!is.null(ga)) ga[k, ],
      num_nodes = length(nodes),
      directed = batch$directed,
      node_ids = batch$original_node_ids[[k]],
      node_type = take_rows(batch$node_type, nodes),
      edge_type = take_rows(batch$edge_type, edges),
      edge_pair = batch$edge_pair[edges] - batch$pair_ptr[k],
      train_mask = take_rows(batch$train_mask, nodes),
      val_mask = take_rows(batch$val_mask, nodes),
      test_mask = take_rows(batch$test_mask, nodes),
      metadata = batch$metadata$graphs[[k]] %||% list()
    )
  })
}

#' @export
print.graph_batch <- function(x, ...) {
  cat(sprintf("<graph_batch> %d graphs, %d nodes, %d edges\n", x$num_graphs,
              x$num_nodes, num_edges(x)))
  invisible(x)
}
