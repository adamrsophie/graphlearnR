#' Generic message passing
#'
#' Computes one round of
#' \deqn{h_i' = U\left(h_i, \bigoplus_{j \in N(i)} M(h_i, h_j, e_{ji})\right)}
#' for every target node `i`, where `N(i)` holds the sources of edges that
#' point to `i`. The message function sees whole edge tensors, so the work is
#' vectorized over edges; aggregation uses the [scatter] kernels.
#'
#' The message function receives named arguments
#' `x_i` (target features gathered per edge), `x_j` (source features per
#' edge), `edge_attr`, `edge_weight`, `index` (target index per edge) and
#' `num_nodes`, and must return one row per edge. Unused arguments can be
#' absorbed with `...`. The update function receives `aggr_out` and `x`
#' (target node features) and returns the new node features.
#'
#' For bipartite or heterogeneous relations pass `x = list(x_src, x_dst)`;
#' `edge_index[1, ]` then indexes rows of `x_src` and `edge_index[2, ]`
#' rows of `x_dst`.
#'
#' Permutation equivariance holds whenever `message` and `update` treat
#' edges and nodes row wise, which is the case for all built in layers.
#'
#' @param x Node feature tensor, or a list `list(x_src, x_dst)`.
#' @param edge_index Long tensor, 2 by E, one based.
#' @param message Message function; the default sends `x_j` (scaled by
#'   `edge_weight` when given).
#' @param aggr Aggregation name or names, see [scatter_aggregate()], or a
#'   function `function(messages, index, dim_size)`.
#' @param update Optional update function.
#' @param edge_attr Optional edge feature tensor (E rows).
#' @param edge_weight Optional edge weight tensor (length E).
#' @param num_nodes Number of target nodes. Defaults to the rows of the
#'   target feature tensor.
#' @return A tensor with one row per target node.
#' @examples
#' if (torch::torch_is_installed()) {
#'   x <- torch::torch_tensor(matrix(1:6, 3, 2), dtype = torch::torch_float())
#'   ei <- torch::torch_tensor(rbind(c(1L, 2L), c(3L, 3L)),
#'                             dtype = torch::torch_long())
#'   message_passing(x, ei, aggr = "sum")
#' }
#' @export
message_passing <- function(x, edge_index, message = NULL, aggr = "sum",
                            update = NULL, edge_attr = NULL,
                            edge_weight = NULL, num_nodes = NULL) {
  if (is.list(x) && !inherits(x, "torch_tensor")) {
    x_src <- x[[1L]]
    x_dst <- x[[2L]]
  } else {
    x_src <- x_dst <- x
  }
  num_nodes <- num_nodes %||% x_dst$size(1L)
  src <- edge_index[1L, ]
  dst <- edge_index[2L, ]
  x_j <- x_src$index_select(1L, src)
  x_i <- if (is.null(x_dst)) NULL else x_dst$index_select(1L, dst)
  msg <- if (is.null(message)) {
    if (is.null(edge_weight)) x_j else x_j * edge_weight$unsqueeze(-1L)
  } else {
    message(x_i = x_i, x_j = x_j, edge_attr = edge_attr,
            edge_weight = edge_weight, index = dst, num_nodes = num_nodes)
  }
  out <- if (is.function(aggr)) aggr(msg, dst, num_nodes)
         else scatter_aggregate(msg, dst, num_nodes, aggr)
  if (is.null(update)) out else update(aggr_out = out, x = x_dst)
}

#' Self loop and normalization helpers
#'
#' Tensor level helpers used by the layers. `add_self_loops_tensor()` appends
#' one self loop per node; with `remaining = TRUE` nodes that already have a
#' self loop keep theirs and get no second one. `gcn_norm()` computes the
#' symmetric normalization
#' \eqn{w_{ij} / \sqrt{d_i d_j}} with \eqn{d} the weighted in degree after
#' adding self loops, as in Kipf and Welling (2017).
#'
#' @param edge_index Long tensor, 2 by E.
#' @param num_nodes Number of nodes.
#' @param edge_weight Optional weight tensor of length E.
#' @param fill_value Weight given to the new self loops.
#' @param remaining Only add loops to nodes that do not have one.
#' @param edge_attr Optional edge feature tensor; new loops get the mean
#'   edge feature of their incoming edges (zero if none).
#' @return A list with `edge_index`, `edge_weight` and `edge_attr`.
#' @export
add_self_loops_tensor <- function(edge_index, num_nodes, edge_weight = NULL,
                                  fill_value = 1, remaining = FALSE,
                                  edge_attr = NULL) {
  dev <- edge_index$device
  nodes <- torch_arange(1L, num_nodes, dtype = torch_long(), device = dev)
  if (remaining && edge_index$size(2L) > 0L) {
    loop <- edge_index[1L, ] == edge_index[2L, ]
    has <- scatter_count(edge_index[1L, ][loop], num_nodes) > 0
    nodes <- nodes[has$logical_not()]
  }
  k <- nodes$numel()
  loops <- torch_stack(list(nodes, nodes))
  out <- list(edge_index = torch_cat(list(edge_index, loops), dim = 2L))
  if (!is.null(edge_weight)) {
    fill <- torch_full(k, fill_value, dtype = edge_weight$dtype, device = dev)
    out$edge_weight <- torch_cat(list(edge_weight, fill))
  }
  if (!is.null(edge_attr)) {
    mean_attr <- scatter_mean(edge_attr, edge_index[2L, ], num_nodes)
    out$edge_attr <- torch_cat(list(edge_attr, mean_attr$index_select(1L, nodes)), dim = 1L)
  }
  out
}

#' @rdname add_self_loops_tensor
#' @export
remove_self_loops_tensor <- function(edge_index, edge_weight = NULL, edge_attr = NULL) {
  if (edge_index$size(2L) == 0L) {
    return(list(edge_index = edge_index, edge_weight = edge_weight, edge_attr = edge_attr))
  }
  keep <- edge_index[1L, ] != edge_index[2L, ]
  list(
    edge_index = edge_index[, keep],
    edge_weight = if (!is.null(edge_weight)) edge_weight[keep],
    edge_attr = if (!is.null(edge_attr)) edge_attr[keep, ]
  )
}

#' @rdname add_self_loops_tensor
#' @param add_self_loops Add self loops before normalizing.
#' @param improved Use fill value 2 for self loops (Kipf and Welling's
#'   "improved" variant).
#' @export
gcn_norm <- function(edge_index, num_nodes, edge_weight = NULL,
                     add_self_loops = TRUE, improved = FALSE) {
  dev <- edge_index$device
  if (is.null(edge_weight)) {
    edge_weight <- torch_ones(edge_index$size(2L), device = dev)
  }
  if (add_self_loops) {
    sl <- add_self_loops_tensor(edge_index, num_nodes, edge_weight,
                                fill_value = if (improved) 2 else 1,
                                remaining = TRUE)
    edge_index <- sl$edge_index
    edge_weight <- sl$edge_weight
  }
  deg <- scatter_sum(edge_weight, edge_index[2L, ], num_nodes)
  dinv <- deg$pow(-0.5)
  dinv <- dinv$masked_fill(torch_isinf(dinv), 0)
  w <- dinv$index_select(1L, edge_index[1L, ]) * edge_weight *
    dinv$index_select(1L, edge_index[2L, ])
  list(edge_index = edge_index, edge_weight = w)
}
