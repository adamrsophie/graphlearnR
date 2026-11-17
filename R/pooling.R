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
#' @examples
#' if (torch::torch_is_installed()) {
#'   pool <- global_attention_pool(4)
#'   x <- torch::torch_randn(5, 4)
#'   pool(x, batch = c(1, 1, 2, 2, 2), num_graphs = 2)$shape
#' }
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
#' each graph are kept (a product within rounding error of a whole number
#' counts as that number, so `ratio = 0.28` keeps 7 of 25 nodes), their features are scaled by \eqn{\tanh(y)} and the
#' edges between kept nodes are retained.
#'
#' @param in_features Node feature size.
#' @param ratio Fraction of nodes kept per graph.
#' @return An `nn_module` returning a list with `x`, `edge_index`, `batch`,
#'   `perm` (kept node indices in the input) and `score`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   pool <- topk_pool(2, ratio = 0.5)
#'   pool
#' }
#' @export
topk_pool <- nn_module(
  "topk_pool",
  initialize = function(in_features, ratio = 0.5) {
    if (ratio <= 0 || ratio > 1) stopf("`ratio` must be in (0, 1].")
    self$ratio <- ratio
    self$weight <- nn_parameter(torch_randn(in_features) / sqrt(in_features))
  },
  forward = function(x, edge_index, batch = NULL) {
    score <- torch_matmul(x, self$weight) / self$weight$norm()
    topk_select(x, score, edge_index, batch, self$ratio)
  }
)

#' @rdname topk_pool
#' @param gnn For `sag_pool()`, the layer that scores nodes: `"graph_conv"`,
#'   `"gcn"` or `"sage"`, or a module mapping `(x, edge_index)` to one score
#'   per node.
#' @details `sag_pool()` is self attention graph pooling (SAGPool, Lee, Lee
#'   and Kang 2019): the scores come from a graph convolution,
#'   \eqn{y = \mathrm{GNN}(X, A)}, so they depend on the neighborhood; the
#'   rest is as in `topk_pool()`.
#' @export
sag_pool <- nn_module(
  "sag_pool",
  initialize = function(in_features, ratio = 0.5, gnn = "graph_conv") {
    if (ratio <= 0 || ratio > 1) stopf("`ratio` must be in (0, 1].")
    self$ratio <- ratio
    self$gnn <- if (is.character(gnn)) {
      switch(match.arg(gnn, c("graph_conv", "gcn", "sage")),
        graph_conv = graph_conv(in_features, 1L),
        gcn = gcn_conv(in_features, 1L),
        sage = sage_conv(in_features, 1L)
      )
    } else {
      gnn
    }
  },
  forward = function(x, edge_index, batch = NULL) {
    score <- self$gnn(x, edge_index)$view(-1L)
    topk_select(x, score, edge_index, batch, self$ratio)
  }
)

# Keep the top ceiling(ratio * n_g) nodes of each graph by score. Selection
# and edge filtering run on the device; only the kept indices (`perm`) are
# copied back to R. Within a graph, ties keep node order.
topk_select <- function(x, score, edge_index, batch, ratio) {
  n <- x$size(1L)
  dev <- x$device
  a <- pool_args(x, batch, NULL)
  b <- a$batch
  sd <- score$detach()
  # Score descending, then a stable sort by graph: nodes grouped by graph,
  # best first, ties in node order.
  o1 <- torch_sort(sd, descending = TRUE, stable = TRUE)[[2L]]
  o2 <- torch_sort(b$index_select(1L, o1), stable = TRUE)[[2L]]
  order_t <- o1$index_select(1L, o2)
  bo <- b$index_select(1L, order_t)
  cnt <- scatter_count(b, a$num_graphs, dtype = torch_long())
  start <- torch_cumsum(cnt, dim = 1L) - cnt
  rank <- torch_arange(1L, n, dtype = torch_long(), device = dev) - start$index_select(1L, bo)
  # ceiling(ratio * n) per graph in R double precision (one host transfer of
  # the graph sizes; some backends, such as MPS, have no float64). Products
  # within rounding error of a whole number (0.28 * 25) are snapped to it,
  # so a decimal ratio keeps its exact share.
  kk <- as.numeric(as_r_array(cnt)) * ratio
  k <- torch_tensor(as.integer(ceiling(round(kk, 9))),
                    dtype = torch_long(), device = dev)
  perm_t <- order_t[rank <= k$index_select(1L, bo)]
  xs <- x$index_select(1L, perm_t) * torch_tanh(score$index_select(1L, perm_t))$unsqueeze(2L)
  m <- perm_t$numel()
  new_id <- torch_zeros(n, dtype = torch_long(), device = dev)
  if (m) new_id$index_put_(list(perm_t), torch_arange(1L, m, dtype = torch_long(), device = dev))
  ei_new <- torch_empty(2L, 0L, dtype = torch_long(), device = dev)
  if (edge_index$size(2L)) {
    nid <- torch_stack(list(new_id$index_select(1L, edge_index[1L, ]),
                            new_id$index_select(1L, edge_index[2L, ])))
    keep <- (nid[1L, ] > 0L) & (nid[2L, ] > 0L)
    if (as.logical(keep$any()$item())) ei_new <- nid[, keep]
  }
  list(
    x = xs,
    edge_index = ei_new,
    batch = b$index_select(1L, perm_t),
    perm = as.integer(as_r_array(perm_t)),
    score = score
  )
}

#' Dense adjacency and differentiable pooling
#'
#' `to_dense_adj()` turns a batched `edge_index` into a `B` by `L` by `L`
#' tensor, `L` the largest graph, with `adj[b, i, j]` the weight of the edge
#' from local node `j` to local node `i` of graph `b` (rows are targets, so
#' `adj %*% x` aggregates incoming messages). Repeated edges add up.
#'
#' `dense_diff_pool()` is the DiffPool operator (Ying et al. 2018). With
#' assignment logits `s` (`B` by `L` by `C`), \eqn{S = \mathrm{softmax}(s)}
#' over clusters, padding rows set to zero,
#' \deqn{X' = S^\top X,\quad A' = S^\top A S,}
#' the auxiliary link loss \eqn{\lVert A - S S^\top \rVert_F / (B L^2)} and
#' entropy loss \eqn{\mathrm{mean}_i(-\sum_c S_{ic} \log S_{ic})}.
#'
#' `dense_graph_conv()` is \eqn{W_1 \tilde A X + W_2 X} on dense batches. With
#' `normalize = TRUE`, as PyG's `DenseGraphConv(aggr = "mean")`, each row of
#' \eqn{A X} is divided by \eqn{\max(d_i, 1)}, \eqn{d_i} the row sum of
#' \eqn{A}. For 0/1 adjacency this is the mean over neighbors; rows whose
#' weighted degree is below 1 (common on coarsened graphs) are not scaled up,
#' so they keep the weighted sum. It is the GNN used on coarsened graphs.
#'
#' Memory is \eqn{O(B L^2)}, so these suit batches of small graphs.
#'
#' @param edge_index Long tensor, 2 by E, 1-based.
#' @param batch Graph membership of each node (R vector or tensor).
#' @param edge_weight Optional edge weights.
#' @param num_graphs Number of graphs; defaults to the largest membership.
#' @param x Dense node features, `B` by `L` by `F`.
#' @param adj Dense adjacency, `B` by `L` by `L`.
#' @param s Assignment logits, `B` by `L` by `C`.
#' @param mask Logical `B` by `L` tensor of real (non padding) nodes.
#' @return `to_dense_adj()`: a tensor. `dense_diff_pool()`: a list with `x`,
#'   `adj`, `s` (the softmax assignment), `link_loss` and `entropy_loss`.
#' @references Ying, You, Morris, Ren, Hamilton and Leskovec (2018)
#'   Hierarchical graph representation learning with differentiable pooling.
#' @examples
#' if (torch::torch_is_installed()) {
#'   b <- graph_batch(dataset_synthetic_graphs(3, seed = 1))
#'   tt <- graph_tensors(b)
#'   adj <- to_dense_adj(tt$edge_index, b$batch)
#'   d <- to_dense_batch(tt$x, b$batch)
#'   pooled <- dense_diff_pool(d$x, adj, torch::torch_randn(3, d$L, 2), d$mask)
#'   dim(pooled$adj)
#' }
#' @export
to_dense_adj <- function(edge_index, batch, edge_weight = NULL, num_graphs = NULL) {
  dev <- edge_index$device
  if (!inherits(batch, "torch_tensor")) batch <- torch_tensor(as.integer(batch), dtype = torch_long(), device = dev)
  batch <- batch$to(device = dev)
  # Index arithmetic stays in torch; only the two sizes B and L come back to R.
  B <- num_graphs %||% if (batch$numel()) as.integer(batch$max()$item()) else 0L
  cnt <- scatter_count(batch, B, dtype = torch_long())
  L <- if (B > 0L) max(as.integer(cnt$max()$item()), 1L) else 1L
  start <- torch_cumsum(cnt, dim = 1L) - cnt
  pos <- torch_arange(0, batch$numel() - 1, dtype = torch_long(), device = dev) - start$index_select(1L, batch)
  w <- edge_weight %||% torch_ones(edge_index$size(2L), device = dev)
  out <- torch_zeros(B * L * L, dtype = w$dtype, device = dev)
  if (edge_index$size(2L)) {
    src <- edge_index[1L, ]
    dst <- edge_index[2L, ]
    flat <- (batch$index_select(1L, dst) - 1L) * L * L + pos$index_select(1L, dst) * L +
      pos$index_select(1L, src) + 1L
    out <- out$index_add(1L, flat, w)
  }
  out$view(c(B, L, L))
}

#' @rdname to_dense_adj
#' @export
dense_diff_pool <- function(x, adj, s, mask = NULL) {
  s <- nnf_softmax(s, dim = -1L)
  if (!is.null(mask)) {
    m <- mask$unsqueeze(-1L)$to(dtype = x$dtype)
    x <- x * m
    s <- s * m
  }
  st <- s$transpose(2L, 3L)
  out_x <- torch_matmul(st, x)
  out_adj <- torch_matmul(torch_matmul(st, adj), s)
  link <- (adj - torch_matmul(s, st))$norm(p = 2) / max(adj$numel(), 1)
  ent <- (-s * torch_log(s + 1e-15))$sum(dim = -1L)
  ent <- if (is.null(mask)) ent$sum() / max(ent$numel(), 1) else ent$sum() / mask$sum()$clamp(min = 1)
  list(x = out_x, adj = out_adj, s = s, link_loss = link, entropy_loss = ent)
}

#' @rdname to_dense_adj
#' @param in_features,out_features Feature sizes.
#' @param normalize Divide aggregated messages by the weighted in degree.
#' @export
dense_graph_conv <- nn_module(
  "dense_graph_conv",
  initialize = function(in_features, out_features, normalize = TRUE) {
    self$lin_rel <- nn_linear(in_features, out_features)
    self$lin_root <- nn_linear(in_features, out_features, bias = FALSE)
    self$normalize <- normalize
  },
  forward = function(x, adj, mask = NULL) {
    agg <- torch_matmul(adj, x)
    if (self$normalize) agg <- agg / adj$sum(dim = -1L, keepdim = TRUE)$clamp(min = 1)
    out <- self$lin_rel(agg) + self$lin_root(x)
    if (!is.null(mask)) out <- out * mask$unsqueeze(-1L)$to(dtype = out$dtype)
    out
  }
)

#' DiffPool graph classification model
#'
#' Two [graph_conv()] stacks (sum aggregation) compute node embeddings and
#' soft cluster assignments
#' on the sparse batch; [dense_diff_pool()] coarsens each graph to
#' `num_clusters` nodes; a [dense_graph_conv()] runs on the coarsened graphs;
#' the cluster embeddings are averaged and passed to a linear head. After
#' each forward pass the sum of the link and entropy losses is stored in
#' `model$aux_loss`, which [train_graph_model()] adds to the task loss.
#'
#' @param input_dim Node feature size.
#' @param hidden_dim Hidden size.
#' @param output_dim Output size (classes or targets).
#' @param num_clusters Clusters per graph after pooling.
#' @param num_layers Message passing layers before pooling.
#' @param aux_weight Weight of the auxiliary losses.
#' @return An `nn_module`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   gs <- dataset_synthetic_graphs(40, seed = 1)
#'   m <- diffpool_model(1, 16, 2, num_clusters = 4)
#'   fit <- train_graph_model(m, gs, "graph_classification", epochs = 3, seed = 1)
#' }
#' @export
diffpool_model <- nn_module(
  "diffpool_model",
  initialize = function(input_dim, hidden_dim = 32L, output_dim = 2L, num_clusters = 8L,
                        num_layers = 2L, aux_weight = 1) {
    mk <- function(out_last) nn_module_list(lapply(seq_len(num_layers), function(i) {
      graph_conv(if (i == 1L) input_dim else hidden_dim, if (i == num_layers) out_last else hidden_dim)
    }))
    self$embed <- mk(hidden_dim)
    self$assign <- mk(num_clusters)
    self$post <- dense_graph_conv(hidden_dim, hidden_dim)
    self$head <- nn_linear(hidden_dim, output_dim)
    self$aux_weight <- aux_weight
    self$config <- list(architecture = "diffpool", input_dim = input_dim, hidden_dim = hidden_dim,
                        output_dim = output_dim, num_clusters = num_clusters,
                        num_layers = num_layers)
    self$aux_loss <- NULL
  },
  forward = function(x, edge_index, batch = NULL, num_graphs = NULL, return_embedding = FALSE,
                     edge_weight = NULL, ...) {
    batch <- batch %||% torch_ones(x$size(1L), dtype = torch_long(), device = x$device)
    # One row per graph, including trailing graphs without nodes.
    num_graphs <- num_graphs %||% if (batch$numel()) as.integer(batch$max()$item()) else 1L
    run <- function(layers, h, act_last) {
      for (i in seq_along(layers)) {
        h <- layers[[i]](h, edge_index, edge_weight = edge_weight)
        if (i < length(layers) || act_last) h <- nnf_relu(h)
      }
      h
    }
    z <- run(self$embed, x, TRUE)
    s <- run(self$assign, x, FALSE)
    dz <- to_dense_batch(z, batch, num_graphs)
    ds <- to_dense_batch(s, batch, num_graphs)
    adj <- to_dense_adj(edge_index, batch, edge_weight, num_graphs = num_graphs)
    pooled <- dense_diff_pool(dz$x, adj, ds$x, dz$mask)
    self$aux_loss <- self$aux_weight * (pooled$link_loss + pooled$entropy_loss)
    h <- nnf_relu(self$post(pooled$x, pooled$adj))$mean(dim = 2L)
    if (return_embedding) h else self$head(h)
  }
)
