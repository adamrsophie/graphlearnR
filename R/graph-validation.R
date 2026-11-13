#' Validate a graph data object
#'
#' Checks that every stored field is consistent with the number of nodes and
#' the number of stored edges: edge indices are whole numbers in
#' `1:num_nodes`, feature matrices have one row per node or edge, masks are
#' logical vectors of the right length, and labels have one entry per node or
#' edge. Problems are reported together in one error.
#'
#' Self loops, duplicate edges (multigraphs) and isolated nodes are valid;
#' query them with [has_self_loops()], [has_duplicate_edges()] and
#' [has_isolated_nodes()].
#'
#' @param g A `graph_data` object.
#' @return `g`, invisibly, if it is valid.
#' @examples
#' g <- graph_data(edge_index = rbind(1:2, 2:3))
#' validate_graph(g)
#' try(graph_data(edge_index = rbind(1, 5), num_nodes = 3))
#' @export
validate_graph <- function(g) {
  if (!inherits(g, "graph_data")) {
    stopf("Expected a `graph_data` object.")
  }
  n <- g$num_nodes
  ei <- g$edge_index
  e <- ncol(ei)
  problems <- character()
  add <- function(...) problems <<- c(problems, sprintf(...))

  if (!is.matrix(ei) || nrow(ei) != 2L) {
    add("`edge_index` must be a 2 by E integer matrix.")
  } else if (e > 0L) {
    bad <- ei < 1L | ei > n
    if (any(bad)) {
      cols <- which(colSums(bad) > 0L)
      add("`edge_index` refers to nodes outside 1:%d (first bad edge: column %d, %d -> %d).",
          n, cols[1L], ei[1L, cols[1L]], ei[2L, cols[1L]])
    }
  }
  check_rows <- function(v, name, expected, what) {
    if (!is.null(v) && n_rows(v) != expected) {
      add("`%s` has %d entries but the graph has %d %s.", name, n_rows(v),
          expected, what)
    }
  }
  check_rows(g$x, "x", n, "nodes")
  check_rows(g$y, "y", n, "nodes")
  check_rows(g$node_ids, "node_ids", n, "nodes")
  check_rows(g$node_type, "node_type", n, "nodes")
  check_rows(g$edge_attr, "edge_attr", e, "stored edges")
  check_rows(g$edge_weight, "edge_weight", e, "stored edges")
  check_rows(g$edge_y, "edge_y", e, "stored edges")
  check_rows(g$edge_type, "edge_type", e, "stored edges")
  check_rows(g$edge_pair, "edge_pair", e, "stored edges")
  if (!is.null(g$node_ids) && anyDuplicated(g$node_ids)) {
    add("`node_ids` must be unique.")
  }
  if (!is.null(g$x) && anyNA(g$x)) {
    add("`x` contains missing values; impute or drop them first.")
  }
  if (!is.null(g$edge_attr) && anyNA(g$edge_attr)) {
    add("`edge_attr` contains missing values.")
  }
  if (!is.null(g$edge_weight) && any(!is.finite(g$edge_weight))) {
    add("`edge_weight` must be finite.")
  }
  for (m in c("train_mask", "val_mask", "test_mask")) {
    v <- g[[m]]
    if (is.null(v)) next
    if (!is.logical(v) || length(v) != n || anyNA(v)) {
      add("`%s` must be a logical vector of length %d without NA.", m, n)
    }
  }
  if (!is.null(g$train_mask) && !is.null(g$test_mask) &&
      length(g$train_mask) == length(g$test_mask) &&
      any(g$train_mask & g$test_mask)) {
    add("`train_mask` and `test_mask` overlap.")
  }
  if (!is.null(g$y) && !is.factor(g$y) && !is.numeric(g$y) && !is.logical(g$y)) {
    add("`y` must be a factor, numeric or logical.")
  }
  if (length(problems)) {
    stop(paste(c("Invalid graph_data:", paste0("* ", problems)),
               collapse = "\n"), call. = FALSE)
  }
  invisible(g)
}
