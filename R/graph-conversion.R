#' Convert between igraph and graph_data
#'
#' `from_igraph()` builds a [graph_data()] object from an `igraph` graph.
#' Vertex order is kept, so vertex `i` becomes node `i`. Vertex names (or a
#' `node_id` vertex attribute written by `to_igraph()`) become `node_ids`.
#' Numeric vertex attributes named in `x_attrs` become node feature columns
#' in that order; edge attributes named in `edge_attrs` become edge feature
#' columns.
#'
#' `to_igraph()` writes features as attributes `x_1`, `x_2`, ... on vertices
#' and `e_1`, `e_2`, ... on edges, labels as `y` and `edge_y`, weights as
#' `weight`, and the original identifiers as `node_id`. With the default
#' arguments, `from_igraph()` reads these back, so the round trip keeps node
#' order, identifiers, features, labels, weights, types and direction.
#'
#' For undirected graphs `to_igraph()` writes each undirected pair once.
#'
#' @param ig An `igraph` object.
#' @param x_attrs Character vector of numeric vertex attributes to use as
#'   node features. `NULL` picks attributes named `x_1`, `x_2`, ... if any.
#' @param y_attr Vertex attribute holding node labels (`NULL` for none). The
#'   default uses `y` when present.
#' @param edge_attrs Edge attributes to use as edge features. `NULL` picks
#'   `e_1`, `e_2`, ... if any.
#' @param weight_attr Edge attribute holding weights, used when present.
#' @param edge_y_attr Edge attribute holding edge labels, used when present.
#' @param g A `graph_data` object.
#' @return `from_igraph()` returns a `graph_data` object; `to_igraph()`
#'   returns an `igraph` object.
#' @examples
#' ig <- igraph::make_ring(5)
#' g <- from_igraph(ig)
#' g
#' ig2 <- to_igraph(g)
#' igraph::ecount(ig2)
#' @export
from_igraph <- function(ig, x_attrs = NULL, y_attr = "y", edge_attrs = NULL,
                        weight_attr = "weight", edge_y_attr = "edge_y") {
  if (!inherits(ig, "igraph")) {
    stopf("`ig` must be an igraph object.")
  }
  n <- igraph::vcount(ig)
  directed <- igraph::is_directed(ig)
  el <- igraph::as_edgelist(ig, names = FALSE)
  vattr <- igraph::vertex_attr_names(ig)
  eattr <- igraph::edge_attr_names(ig)

  numbered <- function(names, prefix) {
    hits <- grep(paste0("^", prefix, "_[0-9]+$"), names, value = TRUE)
    hits[order(as.integer(sub(paste0("^", prefix, "_"), "", hits)))]
  }
  x_attrs <- x_attrs %||% numbered(vattr, "x")
  edge_attrs <- edge_attrs %||% numbered(eattr, "e")
  get_cols <- function(attrs, fun) {
    if (!length(attrs)) {
      return(NULL)
    }
    missing <- setdiff(attrs, if (identical(fun, igraph::vertex_attr)) vattr else eattr)
    if (length(missing)) {
      stopf("Attributes not found: %s.", paste(missing, collapse = ", "))
    }
    m <- vapply(attrs, function(a) as.double(fun(ig, a)),
                numeric(if (identical(fun, igraph::vertex_attr)) n else nrow(el)))
    matrix(m, ncol = length(attrs), dimnames = list(NULL, attrs))
  }
  ids <- if ("node_id" %in% vattr) igraph::vertex_attr(ig, "node_id")
         else if ("name" %in% vattr) igraph::vertex_attr(ig, "name")
         else seq_len(n)
  opt_v <- function(a) if (!is.null(a) && a %in% vattr) igraph::vertex_attr(ig, a)
  opt_e <- function(a) if (!is.null(a) && a %in% eattr) igraph::edge_attr(ig, a)
  masks <- lapply(c("train_mask", "val_mask", "test_mask"), opt_v)

  graph_data(
    x = get_cols(x_attrs, igraph::vertex_attr),
    edge_index = if (nrow(el)) t(el) else NULL,
    y = opt_v(y_attr),
    edge_attr = get_cols(edge_attrs, igraph::edge_attr),
    edge_weight = opt_e(weight_attr),
    edge_y = opt_e(edge_y_attr),
    num_nodes = n,
    directed = directed,
    node_ids = ids,
    node_type = opt_v("node_type"),
    edge_type = opt_e("edge_type"),
    train_mask = masks[[1L]], val_mask = masks[[2L]], test_mask = masks[[3L]]
  )
}

#' @rdname from_igraph
#' @export
to_igraph <- function(g) {
  keep <- input_edge_columns(g)
  ei <- g$edge_index[, keep, drop = FALSE]
  ig <- igraph::make_empty_graph(n = g$num_nodes, directed = g$directed)
  if (ncol(ei)) {
    ig <- igraph::add_edges(ig, as.vector(ei))
  }
  ig <- igraph::set_vertex_attr(ig, "node_id", value = g$node_ids)
  ig <- igraph::set_vertex_attr(ig, "name", value = as.character(g$node_ids))
  if (!is.null(g$x)) {
    for (j in seq_len(ncol(g$x))) {
      ig <- igraph::set_vertex_attr(ig, paste0("x_", j), value = g$x[, j])
    }
  }
  for (a in c("y", "node_type", "train_mask", "val_mask", "test_mask")) {
    if (!is.null(g[[a]])) ig <- igraph::set_vertex_attr(ig, a, value = g[[a]])
  }
  if (ncol(ei)) {
    if (!is.null(g$edge_attr)) {
      for (j in seq_len(ncol(g$edge_attr))) {
        ig <- igraph::set_edge_attr(ig, paste0("e_", j),
                                    value = g$edge_attr[keep, j])
      }
    }
    if (!is.null(g$edge_weight)) ig <- igraph::set_edge_attr(ig, "weight", value = g$edge_weight[keep])
    if (!is.null(g$edge_y)) ig <- igraph::set_edge_attr(ig, "edge_y", value = take_rows(g$edge_y, keep))
    if (!is.null(g$edge_type)) ig <- igraph::set_edge_attr(ig, "edge_type", value = g$edge_type[keep])
  }
  ig
}

# Stored edge columns that correspond one to one with input edges: all of
# them for directed graphs, the first direction of each pair otherwise.
input_edge_columns <- function(g) {
  if (g$directed) seq_len(ncol(g$edge_index)) else which(!duplicated(g$edge_pair))
}

#' Build a graph from an edge list
#'
#' Accepts edges that refer to external node identifiers (character or
#' numeric, not necessarily consecutive) and maps them to one based node
#' indices. Node order follows `nodes[[id_col]]` when a node table is given,
#' and otherwise the order of first appearance in `from` then `to`.
#'
#' @param edges Data frame with one row per edge.
#' @param from,to Column names in `edges` holding source and target
#'   identifiers.
#' @param nodes Optional data frame with one row per node, including nodes
#'   without edges.
#' @param id_col Identifier column in `nodes`.
#' @param feature_cols Numeric columns of `nodes` used as node features.
#' @param label_col Column of `nodes` used as node labels.
#' @param edge_feature_cols Numeric columns of `edges` used as edge features.
#' @param weight_col Column of `edges` holding edge weights.
#' @param edge_label_col Column of `edges` holding edge labels.
#' @param directed Whether edges are directed.
#' @return A `graph_data` object.
#' @examples
#' edges <- data.frame(from = c("a", "b", "c"), to = c("b", "c", "a"),
#'                     w = c(0.5, 1, 2))
#' g <- graph_from_edgelist(edges, weight_col = "w")
#' g$node_ids
#' edgelist_from_graph(g)
#' @export
graph_from_edgelist <- function(edges, from = "from", to = "to", nodes = NULL,
                                id_col = "id", feature_cols = NULL,
                                label_col = NULL, edge_feature_cols = NULL,
                                weight_col = NULL, edge_label_col = NULL,
                                directed = TRUE) {
  edges <- as.data.frame(edges)
  for (col in c(from, to, edge_feature_cols, weight_col, edge_label_col)) {
    if (!col %in% names(edges)) stopf("Column `%s` not found in `edges`.", col)
  }
  # Factor identifiers are matched by their labels, not their codes.
  for (col in c(from, to)) if (is.factor(edges[[col]])) edges[[col]] <- as.character(edges[[col]])
  if (anyNA(edges[[from]]) || anyNA(edges[[to]])) {
    stopf("Edge endpoints contain missing values.")
  }
  if (!is.null(nodes)) {
    nodes <- as.data.frame(nodes)
    if (!id_col %in% names(nodes)) stopf("Column `%s` not found in `nodes`.", id_col)
    ids <- nodes[[id_col]]
    if (is.factor(ids)) ids <- as.character(ids)
    if (anyDuplicated(ids)) stopf("Node identifiers in `nodes` are not unique.")
  } else {
    ids <- unique(c(rbind(edges[[from]], edges[[to]])))
  }
  src <- match(edges[[from]], ids)
  dst <- match(edges[[to]], ids)
  if (anyNA(src) || anyNA(dst)) {
    unknown <- unique(c(edges[[from]][is.na(src)], edges[[to]][is.na(dst)]))
    stopf("Edges refer to identifiers missing from `nodes`: %s.",
          paste(head(unknown, 5L), collapse = ", "))
  }
  graph_data(
    x = if (length(feature_cols)) nodes[, feature_cols, drop = FALSE],
    edge_index = rbind(src, dst),
    y = if (!is.null(label_col)) nodes[[label_col]],
    edge_attr = if (length(edge_feature_cols)) edges[, edge_feature_cols, drop = FALSE],
    edge_weight = if (!is.null(weight_col)) edges[[weight_col]],
    edge_y = if (!is.null(edge_label_col)) edges[[edge_label_col]],
    num_nodes = length(ids),
    directed = directed,
    node_ids = ids
  )
}

#' @rdname graph_from_edgelist
#' @param g A `graph_data` object.
#' @export
edgelist_from_graph <- function(g) {
  keep <- input_edge_columns(g)
  out <- data.frame(from = g$node_ids[g$edge_index[1L, keep]],
                    to = g$node_ids[g$edge_index[2L, keep]])
  if (!is.null(g$edge_weight)) out$weight <- g$edge_weight[keep]
  if (!is.null(g$edge_attr)) {
    ea <- g$edge_attr[keep, , drop = FALSE]
    colnames(ea) <- colnames(ea) %||% paste0("e_", seq_len(ncol(ea)))
    out <- cbind(out, as.data.frame(ea))
  }
  if (!is.null(g$edge_y)) out$edge_y <- take_rows(g$edge_y, keep)
  out
}

#' Build a graph from an adjacency matrix
#'
#' Every nonzero entry `A[i, j]` becomes an edge from node `i` to node `j`
#' with weight `A[i, j]`. For an undirected graph only the upper triangle
#' (including the diagonal) is read, so each pair is created once and stored
#' in both directions. Dense matrices and `Matrix` sparse matrices are
#' accepted.
#'
#' @param A Square numeric matrix or `Matrix` object.
#' @param directed `NULL` treats a symmetric matrix as undirected and any
#'   other matrix as directed.
#' @param weighted Keep entries as edge weights.
#' @param ... Passed to [graph_data()] (for example `x` and `y`).
#' @return A `graph_data` object.
#' @examples
#' A <- matrix(c(0, 1, 0,
#'               1, 0, 2,
#'               0, 2, 0), 3, byrow = TRUE)
#' g <- graph_from_adjacency(A)
#' g$edge_weight
#' as.matrix(adjacency_from_graph(g))
#' @export
graph_from_adjacency <- function(A, directed = NULL, weighted = TRUE, ...) {
  if (!(is.matrix(A) || inherits(A, "Matrix")) || nrow(A) != ncol(A)) {
    stopf("`A` must be a square matrix.")
  }
  n <- nrow(A)
  if (is.null(directed)) {
    directed <- !Matrix::isSymmetric(A)
  }
  # A general numeric sparse matrix lists every stored entry; symmetric and
  # triangular classes store one triangle only, and pattern matrices have no
  # values.
  A <- methods::as(methods::as(methods::as(A, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  trip <- Matrix::summary(methods::as(A, "TsparseMatrix"))
  trip <- trip[trip$x != 0, , drop = FALSE]
  if (!directed) {
    trip <- trip[trip$i <= trip$j, , drop = FALSE]
  }
  trip <- trip[order(trip$i, trip$j), , drop = FALSE]
  graph_data(
    edge_index = if (nrow(trip)) rbind(trip$i, trip$j) else NULL,
    edge_weight = if (weighted && nrow(trip)) trip$x,
    num_nodes = n,
    directed = directed,
    ...
  )
}

#' @rdname graph_from_adjacency
#' @param g A `graph_data` object.
#' @param sparse Return a `Matrix` sparse matrix instead of a dense one.
#' @details `adjacency_from_graph()` sums weights of duplicate edges. Without
#'   weights every edge counts 1.
#' @export
adjacency_from_graph <- function(g, sparse = TRUE, weighted = TRUE) {
  n <- g$num_nodes
  w <- if (weighted && !is.null(g$edge_weight)) g$edge_weight else rep(1, num_edges(g))
  A <- Matrix::sparseMatrix(i = g$edge_index[1L, ], j = g$edge_index[2L, ],
                            x = w, dims = c(n, n))
  if (sparse) A else as.matrix(A)
}

#' Convert graph data to torch tensors
#'
#' Converts the stored R values once, so a training loop can reuse the same
#' tensors on one device. Node indices stay one based, which is what R
#' `torch` index operations expect. Factor and integer labels become `long`
#' tensors (one based class codes, as [torch::nnf_cross_entropy()] expects);
#' numeric labels become floating point tensors.
#'
#' @param g A `graph_data` or `graph_batch` object.
#' @param device A `torch_device`, or one of `"cpu"`, `"cuda"`, `"mps"`,
#'   `"auto"`.
#' @param dtype Floating point dtype for features.
#' @return A list of tensors with class `graph_tensors`: `x`, `edge_index`
#'   (2 by E, long), `edge_weight`, `edge_attr`, `y`, `edge_y`, `graph_y`,
#'   `batch` (graph membership for batches), masks, and `num_nodes`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- graph_data(x = diag(3), edge_index = rbind(1:2, 2:3))
#'   tt <- graph_tensors(g)
#'   tt$edge_index
#' }
#' @export
graph_tensors <- function(g, device = NULL, dtype = torch_float()) {
  dev <- resolve_device(device)
  ft <- function(v) {
    if (is.null(v)) {
      return(NULL)
    }
    if (is.matrix(v)) {
      if (!is.double(v)) storage.mode(v) <- "double"
      torch_tensor(v, dtype = dtype, device = dev)
    } else torch_tensor(as.double(v), dtype = dtype, device = dev)
  }
  lab <- function(v) {
    if (is.null(v)) {
      return(NULL)
    }
    if (is.factor(v) || (is.integer(v) && !is.matrix(v))) {
      torch_tensor(as.integer(v), dtype = torch_long(), device = dev)
    } else if (is.logical(v)) {
      torch_tensor(as.double(v), dtype = dtype, device = dev)
    } else {
      ft(v)
    }
  }
  mask <- function(v) if (!is.null(v)) torch_tensor(v, dtype = torch_bool(), device = dev)
  n <- num_nodes(g)
  out <- list(
    x = ft(g$x),
    edge_index = if (ncol(g$edge_index)) {
      torch_tensor(g$edge_index, dtype = torch_long(), device = dev)
    } else {
      torch_empty(2L, 0L, dtype = torch_long(), device = dev)
    },
    edge_weight = ft(g$edge_weight),
    edge_attr = ft(g$edge_attr),
    y = lab(g$y),
    edge_y = lab(g$edge_y),
    edge_type = if (!is.null(g$edge_type)) {
      torch_tensor(as.integer(g$edge_type), dtype = torch_long(), device = dev)
    },
    graph_y = lab(g$graph_y),
    graph_attr = if (!is.null(g$graph_attr)) {
      ga <- g$graph_attr
      torch_tensor(if (is.matrix(ga)) ga else matrix(ga, nrow = 1L), dtype = dtype, device = dev)
    },
    batch = torch_tensor(g$batch %||% rep(1L, n), dtype = torch_long(), device = dev),
    num_graphs = g$num_graphs %||% 1L,
    train_mask = mask(g$train_mask),
    val_mask = mask(g$val_mask),
    test_mask = mask(g$test_mask),
    num_nodes = n,
    device = dev
  )
  structure(out, class = "graph_tensors")
}

#' @export
print.graph_tensors <- function(x, ...) {
  cat(sprintf("<graph_tensors> %d nodes, %d edges on %s\n", x$num_nodes,
              x$edge_index$size(2), x$device$type))
  invisible(x)
}
