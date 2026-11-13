#' Create a graph data object
#'
#' `graph_data()` stores one homogeneous graph: node features, an edge index,
#' optional edge features and weights, labels at node, edge and graph level,
#' node and edge type codes, split masks and free metadata. Data are kept as
#' ordinary R vectors and matrices; [graph_tensors()] turns them into `torch`
#' tensors once, on the device you choose.
#'
#' @section Indexing convention:
#' Node indices are **one based** everywhere in the public API: node `i` is
#' row `i` of `x`, and `edge_index` holds values in `1:num_nodes`. R `torch`
#' index operations (`index_select`, `index_add`, `scatter_add`,
#' `scatter_reduce`) are also one based, so the stored indices are passed to
#' `torch` unchanged. The only conversion to zero based indices happens in
#' [to_zero_based()], used when exporting to formats that require it.
#'
#' @section Edge direction:
#' `edge_index` is a 2 by E matrix; column `j` is a directed edge from
#' `edge_index[1, j]` (source) to `edge_index[2, j]` (target). Messages flow
#' from source to target.
#'
#' With `directed = FALSE`, supply each undirected edge **once**. The stored
#' graph holds both directions, `(u, v)` and `(v, u)`, next to each other; a
#' self loop is stored once. `edge_pair` maps every stored directed edge to the
#' input edge it came from, and edge level values (`edge_attr`,
#' `edge_weight`, `edge_y`, `edge_type`) are copied to both directions. If the
#' input already contains reciprocal pairs, a warning is given, because those
#' pairs would then be stored twice in each direction.
#'
#' @param x Node features: numeric matrix with one row per node, a numeric
#'   vector (one feature), a numeric data frame, or `NULL`.
#' @param edge_index Edges as a 2 by E matrix (row 1 source, row 2 target), an
#'   E by 2 matrix, or a data frame with columns `from` and `to`. Values are
#'   one based node indices. `NULL` means no edges.
#' @param y Node labels: factor (classification), integer or numeric vector,
#'   or numeric matrix (multi target), one entry per node. Character labels
#'   are converted to a factor (as are character `edge_y` and `graph_y`).
#' @param edge_attr Edge features, one row per input edge.
#' @param edge_weight Positive or signed numeric edge weights, one per input
#'   edge.
#' @param edge_y Edge labels, one per input edge.
#' @param graph_y Graph label (scalar, factor of length one, or numeric
#'   vector).
#' @param graph_attr Graph level features (numeric vector).
#' @param num_nodes Number of nodes. Inferred from `x`, `y`, `node_ids` or
#'   the largest edge index when `NULL`.
#' @param directed Whether edges are directed. See the edge direction section.
#' @param node_ids External node identifiers, one per node. Default
#'   `seq_len(num_nodes)`.
#' @param node_type,edge_type Optional integer or factor type codes per node
#'   and per input edge. They are stored, not interpreted; typed learning uses
#'   [hetero_graph_data()].
#' @param train_mask,val_mask,test_mask Logical node masks.
#' @param metadata Named list of free metadata (source, version, notes).
#' @param validate Run [validate_graph()] after construction.
#' @return An object of class `graph_data`.
#' @examples
#' g <- graph_data(
#'   x = matrix(rnorm(8), 4, 2),
#'   edge_index = rbind(c(1, 2, 3), c(2, 3, 4)),
#'   y = factor(c("a", "b", "a", "b")),
#'   directed = FALSE
#' )
#' g
#' num_nodes(g)
#' num_edges(g)
#' num_edges(g, unique_pairs = TRUE)
#' @export
graph_data <- function(x = NULL, edge_index = NULL, y = NULL,
                       edge_attr = NULL, edge_weight = NULL, edge_y = NULL,
                       graph_y = NULL, graph_attr = NULL, num_nodes = NULL,
                       directed = TRUE, node_ids = NULL, node_type = NULL,
                       edge_type = NULL, train_mask = NULL, val_mask = NULL,
                       test_mask = NULL, metadata = list(), validate = TRUE) {
  check_flag(directed, "directed")
  x <- as_feature_matrix(x, "x")
  edge_attr <- as_feature_matrix(edge_attr, "edge_attr")
  ei <- normalize_edge_index(edge_index)

  if (is.null(num_nodes)) {
    candidates <- c(n_rows(x), n_rows(y), length(node_ids) %||% NA,
                    length(node_type) %||% NA)
    candidates <- candidates[!is.na(candidates) & candidates > 0]
    num_nodes <- if (length(candidates)) candidates[1L] else 0L
    if (ncol(ei) > 0L) {
      num_nodes <- max(num_nodes, max(ei, na.rm = TRUE))
    }
  }
  num_nodes <- check_count(num_nodes, "num_nodes")

  n_in <- ncol(ei)
  edge_inputs <- list(edge_attr = edge_attr, edge_weight = edge_weight,
                      edge_y = edge_y, edge_type = edge_type)
  for (nm in names(edge_inputs)) {
    v <- edge_inputs[[nm]]
    if (!is.null(v) && n_rows(v) != n_in) {
      stopf("`%s` has %d entries but there are %d input edges.", nm,
            n_rows(v), n_in)
    }
  }
  # Character labels become factors, so class levels are explicit.
  as_label <- function(v) if (is.character(v)) factor(v) else v
  y <- as_label(y)
  edge_y <- as_label(edge_y)
  graph_y <- as_label(graph_y)
  edge_pair <- seq_len(n_in)
  if (!directed && n_in > 0L) {
    warn_reciprocal(ei)
    loop <- ei[1L, ] == ei[2L, ]
    reps <- ifelse(loop, 1L, 2L)
    edge_pair <- rep(seq_len(n_in), reps)
    first <- !duplicated(edge_pair)
    src <- ifelse(first, ei[1L, edge_pair], ei[2L, edge_pair])
    dst <- ifelse(first, ei[2L, edge_pair], ei[1L, edge_pair])
    ei <- rbind(as.integer(src), as.integer(dst))
  }
  expand <- function(v) if (is.null(v) || directed) v else take_rows(v, edge_pair)

  g <- new_graph_data(
    x = x, edge_index = ei, y = y,
    edge_attr = expand(edge_attr), edge_weight = expand(edge_weight),
    edge_y = expand(edge_y), graph_y = graph_y, graph_attr = graph_attr,
    num_nodes = num_nodes, directed = directed,
    node_ids = node_ids %||% seq_len(num_nodes),
    node_type = node_type, edge_type = expand(edge_type),
    edge_pair = edge_pair, train_mask = train_mask, val_mask = val_mask,
    test_mask = test_mask, metadata = metadata
  )
  if (validate) validate_graph(g)
  g
}

# Low level constructor: takes the stored (already expanded) form.
new_graph_data <- function(x = NULL, edge_index = matrix(integer(), 2L, 0L),
                           y = NULL, edge_attr = NULL, edge_weight = NULL,
                           edge_y = NULL, graph_y = NULL, graph_attr = NULL,
                           num_nodes = 0L, directed = TRUE,
                           node_ids = seq_len(num_nodes), node_type = NULL,
                           edge_type = NULL, edge_pair = seq_len(ncol(edge_index)),
                           train_mask = NULL, val_mask = NULL, test_mask = NULL,
                           metadata = list(), class = character()) {
  storage.mode(edge_index) <- "integer"
  structure(
    list(
      x = x, edge_index = edge_index, edge_attr = edge_attr,
      edge_weight = if (is.null(edge_weight)) NULL else as.double(edge_weight),
      y = y, edge_y = edge_y, graph_y = graph_y,
      graph_attr = if (is.null(graph_attr)) NULL else as.double(graph_attr),
      num_nodes = as.integer(num_nodes), directed = directed,
      node_ids = node_ids, node_type = node_type, edge_type = edge_type,
      edge_pair = as.integer(edge_pair),
      train_mask = train_mask, val_mask = val_mask, test_mask = test_mask,
      metadata = metadata
    ),
    class = c(class, "graph_data")
  )
}

normalize_edge_index <- function(edge_index) {
  if (is.null(edge_index)) {
    return(matrix(integer(), 2L, 0L))
  }
  if (inherits(edge_index, "torch_tensor")) {
    edge_index <- as.array(edge_index$cpu())
  }
  if (is.data.frame(edge_index)) {
    if (!all(c("from", "to") %in% names(edge_index))) {
      stopf("An edge data frame needs columns `from` and `to`.")
    }
    edge_index <- rbind(edge_index$from, edge_index$to)
  }
  if (!is.matrix(edge_index)) {
    stopf("`edge_index` must be a 2 by E matrix, an E by 2 matrix or a data frame.")
  }
  if (nrow(edge_index) != 2L) {
    if (ncol(edge_index) == 2L) {
      edge_index <- t(edge_index)
    } else {
      stopf("`edge_index` must have 2 rows (or 2 columns); it is %d by %d.",
            nrow(edge_index), ncol(edge_index))
    }
  }
  if (!is.numeric(edge_index) || !is_integerish(edge_index)) {
    stopf("`edge_index` must hold whole numbers (one based node indices).")
  }
  if (anyNA(edge_index)) {
    stopf("`edge_index` contains missing values.")
  }
  dimnames(edge_index) <- NULL
  storage.mode(edge_index) <- "integer"
  edge_index
}

warn_reciprocal <- function(ei) {
  nl <- ei[1L, ] != ei[2L, ]
  if (!any(nl)) {
    return(invisible())
  }
  fwd <- paste(ei[1L, nl], ei[2L, nl])
  rev <- paste(ei[2L, nl], ei[1L, nl])
  if (any(rev %in% fwd)) {
    warnf(paste(
      "Undirected input contains reciprocal pairs (both u-v and v-u).",
      "Supply each undirected edge once, or use `directed = TRUE` if the",
      "list already holds both directions."
    ))
  }
  invisible()
}

#' Basic graph queries
#'
#' @param g A `graph_data`, `graph_batch` or `hetero_graph_data` object.
#' @param unique_pairs For undirected graphs, count each undirected edge once
#'   instead of counting stored directed entries.
#' @return An integer (or logical) scalar; `degree()` returns an integer
#'   vector.
#' @name graph-queries
#' @examples
#' g <- graph_data(edge_index = rbind(c(1, 2), c(2, 3)), directed = FALSE)
#' num_nodes(g)
#' num_edges(g)
#' num_edges(g, unique_pairs = TRUE)
#' degree(g)
NULL

#' @rdname graph-queries
#' @export
num_nodes <- function(g) UseMethod("num_nodes")

#' @export
num_nodes.graph_data <- function(g) g$num_nodes

#' @rdname graph-queries
#' @export
num_edges <- function(g, unique_pairs = FALSE) UseMethod("num_edges")

#' @export
num_edges.graph_data <- function(g, unique_pairs = FALSE) {
  if (unique_pairs && !g$directed) {
    return(length(unique(g$edge_pair)))
  }
  ncol(g$edge_index)
}

#' @rdname graph-queries
#' @export
num_node_features <- function(g) if (is.null(g$x)) 0L else ncol(g$x)

#' @rdname graph-queries
#' @export
num_edge_features <- function(g) if (is.null(g$edge_attr)) 0L else ncol(g$edge_attr)

#' @rdname graph-queries
#' @export
is_directed_graph <- function(g) isTRUE(g$directed)

#' @rdname graph-queries
#' @export
has_self_loops <- function(g) any(g$edge_index[1L, ] == g$edge_index[2L, ])

#' @rdname graph-queries
#' @export
has_isolated_nodes <- function(g) {
  if (g$num_nodes == 0L) {
    return(FALSE)
  }
  used <- logical(g$num_nodes)
  used[c(g$edge_index)] <- TRUE
  !all(used)
}

#' @rdname graph-queries
#' @export
has_duplicate_edges <- function(g) {
  if (ncol(g$edge_index) == 0L) {
    return(FALSE)
  }
  anyDuplicated(paste(g$edge_index[1L, ], g$edge_index[2L, ])) > 0L
}

#' @rdname graph-queries
#' @param mode `"in"` counts incoming edges (edges whose target is the node),
#'   `"out"` counts outgoing edges. For undirected graphs both agree.
#' @param weighted Sum edge weights instead of counting edges.
#' @export
degree <- function(g, mode = c("in", "out"), weighted = FALSE) {
  mode <- match.arg(mode)
  idx <- g$edge_index[if (mode == "in") 2L else 1L, ]
  w <- if (weighted && !is.null(g$edge_weight)) g$edge_weight else rep(1, length(idx))
  out <- numeric(g$num_nodes)
  if (length(idx)) {
    s <- rowsum(w, idx)
    out[as.integer(rownames(s))] <- s[, 1L]
  }
  if (weighted) out else as.integer(out)
}

#' @export
print.graph_data <- function(x, ...) {
  cat(sprintf("<graph_data> %s graph: %d nodes, %d edges%s\n",
              if (x$directed) "directed" else "undirected",
              x$num_nodes, num_edges(x),
              if (x$directed) "" else
                sprintf(" (%d undirected pairs)", num_edges(x, TRUE))))
  fields <- c(
    x = if (!is.null(x$x)) sprintf("node features [%d x %d]", nrow(x$x), ncol(x$x)),
    edge_attr = if (!is.null(x$edge_attr))
      sprintf("edge features [%d x %d]", nrow(x$edge_attr), ncol(x$edge_attr)),
    edge_weight = if (!is.null(x$edge_weight)) "edge weights",
    y = if (!is.null(x$y)) describe_labels(x$y, "node labels"),
    edge_y = if (!is.null(x$edge_y)) describe_labels(x$edge_y, "edge labels"),
    graph_y = if (!is.null(x$graph_y)) describe_labels(x$graph_y, "graph label"),
    masks = mask_summary(x)
  )
  for (f in fields) cat("  ", f, "\n", sep = "")
  invisible(x)
}

describe_labels <- function(y, what) {
  if (is.factor(y)) {
    sprintf("%s: factor with %d levels", what, nlevels(y))
  } else if (is.matrix(y)) {
    sprintf("%s: numeric [%d x %d]", what, nrow(y), ncol(y))
  } else {
    sprintf("%s: %s", what, class(y)[1L])
  }
}

mask_summary <- function(g) {
  m <- c(train = sum(g$train_mask %||% FALSE), val = sum(g$val_mask %||% FALSE),
         test = sum(g$test_mask %||% FALSE))
  if (all(m == 0)) {
    return(NULL)
  }
  paste0("masks: ", paste(names(m), m, sep = "=", collapse = ", "))
}

#' @export
summary.graph_data <- function(object, ...) {
  din <- degree(object, "in")
  out <- list(
    num_nodes = object$num_nodes,
    num_edges = num_edges(object),
    num_pairs = num_edges(object, TRUE),
    directed = object$directed,
    num_node_features = num_node_features(object),
    num_edge_features = num_edge_features(object),
    self_loops = has_self_loops(object),
    isolated_nodes = sum(din == 0L & degree(object, "out") == 0L),
    duplicate_edges = has_duplicate_edges(object),
    in_degree = if (length(din)) summary(din) else NULL,
    label_counts = if (is.factor(object$y)) table(object$y) else NULL
  )
  class(out) <- "summary_graph_data"
  out
}

#' @export
print.summary_graph_data <- function(x, ...) {
  cat(sprintf("Graph summary: %d nodes, %d edges (%s)\n", x$num_nodes,
              x$num_edges, if (x$directed) "directed" else "undirected"))
  cat(sprintf("  node features: %d, edge features: %d\n",
              x$num_node_features, x$num_edge_features))
  cat(sprintf("  self loops: %s, duplicate edges: %s, isolated nodes: %d\n",
              x$self_loops, x$duplicate_edges, x$isolated_nodes))
  if (!is.null(x$in_degree)) {
    cat("  in degree:\n")
    print(x$in_degree)
  }
  if (!is.null(x$label_counts)) {
    cat("  node label counts:\n")
    print(x$label_counts)
  }
  invisible(x)
}

#' Convert one based indices to zero based
#'
#' The single place where graphlearnR converts its one based node indices to
#' the zero based convention used by Python graph libraries and some file
#' formats. [from_zero_based()] is the inverse.
#'
#' @param index Integer vector or matrix of one based indices.
#' @return Integer vector or matrix of the same shape.
#' @examples
#' to_zero_based(rbind(c(1, 2), c(2, 3)))
#' @export
to_zero_based <- function(index) {
  if (anyNA(index) || any(index < 1)) {
    stopf("One based indices must be >= 1 and not missing.")
  }
  out <- index - 1L
  storage.mode(out) <- "integer"
  out
}

#' @rdname to_zero_based
#' @export
from_zero_based <- function(index) {
  if (anyNA(index) || any(index < 0)) {
    stopf("Zero based indices must be >= 0 and not missing.")
  }
  out <- index + 1L
  storage.mode(out) <- "integer"
  out
}
