#' Global pooling (graph readout)
#'
#' Reduce node features to one row per graph using the membership vector
#' `batch` of a [graph_batch()] (values `1:num_graphs`). Sum, mean and max
#' are permutation invariant: reordering nodes within a graph does not change
#' the result.
#'
#' @param x Node feature tensor (N by F).
#' @param batch Long tensor or integer vector of graph membership. `NULL`
#'   treats all nodes as one graph.
#' @param num_graphs Number of graphs; inferred from `batch` when `NULL`.
#' @return A tensor with one row per graph.
#' @examples
#' if (torch::torch_is_installed()) {
#'   x <- torch::torch_tensor(matrix(1:6, 3, 2), dtype = torch::torch_float())
#'   b <- c(1L, 1L, 2L)
#'   global_add_pool(x, b)
#'   global_mean_pool(x, b)
#'   global_max_pool(x, b)
#' }
#' @name global-pooling
NULL

pool_args <- function(x, batch, num_graphs) {
  if (is.null(batch)) {
    batch <- torch_ones(x$size(1L), dtype = torch_long(), device = x$device)
  }
  batch <- as_index_tensor(batch, x$device)
  if (is.null(num_graphs)) {
    num_graphs <- if (batch$numel()) as.integer(batch$max()$cpu()) else 0L
  }
  list(batch = batch, num_graphs = num_graphs)
}

#' @rdname global-pooling
#' @export
global_add_pool <- function(x, batch = NULL, num_graphs = NULL) {
  a <- pool_args(x, batch, num_graphs)
  scatter_sum(x, a$batch, a$num_graphs)
}

#' @rdname global-pooling
#' @export
global_mean_pool <- function(x, batch = NULL, num_graphs = NULL) {
  a <- pool_args(x, batch, num_graphs)
  scatter_mean(x, a$batch, a$num_graphs)
}

#' @rdname global-pooling
#' @export
global_max_pool <- function(x, batch = NULL, num_graphs = NULL) {
  a <- pool_args(x, batch, num_graphs)
  scatter_max(x, a$batch, a$num_graphs)
}

#' Global attention pooling
#'
#' Soft attention readout (Li et al. 2016): each node gets a gate score
#' \eqn{s_i = \mathrm{gate}(x_i)}, scores are normalized with a softmax within
#' each graph, and the graph vector is \eqn{\sum_i \mathrm{softmax}(s)_i\,
#' \mathrm{nn}(x_i)}. Permutation invariant.
#'
#' @param in_features Node feature size.
#' @param gate_nn Module mapping features to one score; default a linear
#'   layer.
#' @param nn Optional module applied to node features before weighting.
#' @return An `nn_module` called as `module(x, batch, num_graphs)`.
#' @export
global_attention_pool <- nn_module(
  "global_attention_pool",
  initialize = function(in_features, gate_nn = NULL, nn = NULL) {
    self$gate_nn <- gate_nn %||% nn_linear(in_features, 1L)
    self$nn <- nn
  },
  forward = function(x, batch = NULL, num_graphs = NULL) {
    a <- pool_args(x, batch, num_graphs)
    gate <- scatter_softmax(self$gate_nn(x)$view(c(-1L, 1L)), a$batch, a$num_graphs)
    self$last_gate <- gate$detach()
    h <- if (is.null(self$nn)) x else self$nn(x)
    scatter_sum(gate * h, a$batch, a$num_graphs)
  }
)

#' Top-k pooling (hierarchical, experimental)
#'
#' Graph U-Net style pooling (Gao and Ji 2019): nodes are scored by
#' \eqn{y = X p / \lVert p \rVert}, the top `ceiling(ratio * n_g)` nodes of
#' each graph are kept, their features are scaled by \eqn{\tanh(y)} and the
#' edges between kept nodes are retained.
#'
#' @param in_features Node feature size.
#' @param ratio Fraction of nodes kept per graph.
#' @return An `nn_module` returning a list with `x`, `edge_index`, `batch`,
#'   `perm` (kept node indices in the input) and `score`.
#' @export
topk_pool <- nn_module(
  "topk_pool",
  initialize = function(in_features, ratio = 0.5) {
    if (ratio <= 0 || ratio > 1) stopf("`ratio` must be in (0, 1].")
    self$ratio <- ratio
    self$weight <- nn_parameter(torch_randn(in_features) / sqrt(in_features))
  },
  forward = function(x, edge_index, batch = NULL) {
    n <- x$size(1L)
    a <- pool_args(x, batch, NULL)
    score <- torch_matmul(x, self$weight) / self$weight$norm()
    b <- as.integer(as_r_array(a$batch))
    s <- as.numeric(as_r_array(score))
    perm <- unlist(lapply(split(seq_len(n), factor(b, levels = seq_len(a$num_graphs))),
                          function(idx) {
      if (!length(idx)) return(integer())
      k <- ceiling(self$ratio * length(idx))
      idx[order(-s[idx])][seq_len(k)]
    }), use.names = FALSE)
    perm_t <- torch_tensor(perm, dtype = torch_long(), device = x$device)
    xs <- x$index_select(1L, perm_t) * torch_tanh(score$index_select(1L, perm_t))$unsqueeze(2L)
    new_id <- integer(n)
    new_id[perm] <- seq_along(perm)
    ei <- as_r_array(edge_index)
    keep <- if (length(ei)) new_id[ei[1L, ]] > 0L & new_id[ei[2L, ]] > 0L else logical()
    ei_new <- if (any(keep)) {
      rbind(new_id[ei[1L, keep]], new_id[ei[2L, keep]])
    } else {
      matrix(integer(), 2L, 0L)
    }
    list(
      x = xs,
      edge_index = if (ncol(ei_new)) torch_tensor(ei_new, dtype = torch_long(), device = x$device)
                   else torch_empty(2L, 0L, dtype = torch_long(), device = x$device),
      batch = torch_tensor(b[perm], dtype = torch_long(), device = x$device),
      perm = perm,
      score = score
    )
  }
)
