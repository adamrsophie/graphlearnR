#' Structural and positional encodings
#'
#' Node level encodings that give attention models information about graph
#' structure.
#'
#' * `laplacian_pe()`: the `k` eigenvectors of the symmetric normalized
#'   Laplacian \eqn{L = I - D^{-1/2} A D^{-1/2}} with the smallest nonzero
#'   eigenvalues (Dwivedi et al. 2020). Direction is ignored (the adjacency is
#'   symmetrized). Eigenvectors are defined only up to sign, and up to a
#'   rotation within repeated eigenvalues, so the encoding is **not**
#'   permutation equivariant as a function: a permuted graph can give
#'   different signs. Train with random sign flips ([random_sign_flip()]) to
#'   make a model robust to this. A graph with `c` connected components
#'   (with edges) has `c` zero eigenvalues, all skipped. Graphs with too few
#'   nonzero eigenvalues are padded with zero columns.
#' * `random_walk_pe()`: the return probabilities
#'   \eqn{[(D^{-1}A)^t]_{ii}} for \eqn{t = 1, \dots, k} (Dwivedi et al.
#'   2022). This encoding is permutation equivariant and has no sign
#'   ambiguity. It forms dense matrix powers, \eqn{O(k n^3)} time.
#' * `shortest_path_distances()`: the all pairs hop distance matrix, with
#'   unreachable pairs set to `max_dist + 1`; used as attention bias by
#'   [graph_transformer_model()]. Equivariant: permuting nodes permutes rows
#'   and columns.
#'
#' @param g A `graph_data` object.
#' @param k Number of encoding dimensions.
#' @param use_rspectra Use `RSpectra` sparse eigensolvers (if installed)
#'   for graphs larger than 500 nodes.
#' @return An N by `k` matrix (`laplacian_pe()`, `random_walk_pe()`) or an N
#'   by N integer matrix.
#' @examples
#' g <- dataset_karate()
#' dim(laplacian_pe(g, 4))
#' dim(random_walk_pe(g, 3))
#' shortest_path_distances(g)[1:3, 1:3]
#' @name positional-encodings
NULL

sym_adjacency <- function(g) {
  A <- adjacency_from_graph(g, weighted = FALSE)
  A <- A + Matrix::t(A)
  A@x[] <- 1
  Matrix::diag(A) <- 0
  Matrix::drop0(A)
}

#' @rdname positional-encodings
#' @export
laplacian_pe <- function(g, k = 8L, use_rspectra = TRUE) {
  n <- g$num_nodes
  if (n == 0L) return(matrix(0, 0L, k))
  A <- sym_adjacency(g)
  d <- Matrix::rowSums(A)
  dinv <- ifelse(d > 0, 1 / sqrt(d), 0)
  L <- Matrix::Diagonal(n) - Matrix::Diagonal(x = dinv) %*% A %*% Matrix::Diagonal(x = dinv)
  # Each connected component with an edge gives one zero eigenvalue (an
  # isolated node has eigenvalue 1 here), so skip that many eigenvectors.
  csize <- igraph::components(igraph::graph_from_adjacency_matrix(A, mode = "undirected"))$csize
  z <- sum(csize > 1L)
  m <- min(k + z, n)
  if (use_rspectra && n > 500L && requireNamespace("RSpectra", quietly = TRUE)) {
    es <- RSpectra::eigs_sym(L, m, which = "SA")
    vals <- es$values
    vecs <- es$vectors
  } else {
    es <- eigen(as.matrix(L), symmetric = TRUE)
    vals <- es$values
    vecs <- es$vectors
  }
  o <- order(vals)
  vecs <- vecs[, o[seq_len(m)], drop = FALSE][, setdiff(seq_len(m), seq_len(z)), drop = FALSE]
  out <- matrix(0, n, k)
  if (ncol(vecs)) out[, seq_len(ncol(vecs))] <- vecs
  out
}

#' @rdname positional-encodings
#' @export
random_walk_pe <- function(g, k = 8L) {
  n <- g$num_nodes
  A <- as.matrix(sym_adjacency(g))
  d <- rowSums(A)
  P <- A / ifelse(d > 0, d, 1)
  out <- matrix(0, n, k)
  M <- diag(n)
  for (t in seq_len(k)) {
    M <- M %*% P
    out[, t] <- diag(M)
  }
  out
}

#' @rdname positional-encodings
#' @param max_dist Distances above this are clipped to `max_dist + 1`.
#' @param directed Follow edge direction.
#' @export
shortest_path_distances <- function(g, max_dist = 10L, directed = FALSE) {
  n <- g$num_nodes
  if (n == 0L) return(matrix(0L, 0L, 0L))
  ig <- igraph::make_empty_graph(n, directed = directed)
  if (ncol(g$edge_index)) ig <- igraph::add_edges(ig, as.vector(g$edge_index))
  D <- igraph::distances(ig, mode = if (directed) "out" else "all")
  D[!is.finite(D) | D > max_dist] <- max_dist + 1
  storage.mode(D) <- "integer"
  D
}

#' @rdname positional-encodings
#' @param pe A positional encoding matrix.
#' @param seed Optional integer seed.
#' @export
random_sign_flip <- function(pe, seed = NULL) {
  s <- with_optional_seed(seed, sample(c(-1, 1), ncol(pe), replace = TRUE))
  sweep(pe, 2L, s, "*")
}

#' @rdname positional-encodings
#' @param type `"laplacian"` or `"random_walk"`.
#' @details `add_positional_encoding()` appends the encoding to the node
#'   features of `g` (or of every graph in a list).
#' @export
add_positional_encoding <- function(g, type = c("random_walk", "laplacian"), k = 8L) {
  type <- match.arg(type)
  if (is.list(g) && !inherits(g, "graph_data")) {
    return(lapply(g, add_positional_encoding, type = type, k = k))
  }
  pe <- if (type == "laplacian") laplacian_pe(g, k) else random_walk_pe(g, k)
  g$x <- if (is.null(g$x)) pe else cbind(g$x, pe)
  g
}

#' Dense batch layout
#'
#' Reorganizes node rows of a batch into a padded `B` by `L` by `F` tensor,
#' where `L` is the largest graph size, plus a `B` by `L` logical mask of
#' real nodes. `from_dense_batch()` inverts it.
#'
#' @param x Node feature tensor (N by F).
#' @param batch Graph membership (integer vector, `1:B`, nodes of a graph
#'   contiguous as produced by [graph_batch()]).
#' @return A list with `x`, `mask` and `index` (the flat positions).
#' @examples
#' if (torch::torch_is_installed()) {
#'   x <- torch::torch_randn(5, 2)
#'   d <- to_dense_batch(x, batch = c(1, 1, 2, 2, 2))
#'   d$x$shape
#'   from_dense_batch(d)$shape
#' }
#' @export
to_dense_batch <- function(x, batch) {
  b <- as.integer(as_r_array(batch))
  B <- if (length(b)) max(b) else 0L
  cnt <- tabulate(b, B)
  L <- max(cnt, 1L)
  pos <- sequence(cnt)
  flat <- (b - 1L) * L + pos
  idx <- torch_tensor(flat, dtype = torch_long(), device = x$device)
  out <- torch_zeros(B * L, x$size(2L), dtype = x$dtype, device = x$device)
  out <- out$index_copy(1L, idx, x)
  mask <- logical(B * L)
  mask[flat] <- TRUE
  list(x = out$view(c(B, L, x$size(2L))),
       mask = torch_tensor(matrix(mask, B, L, byrow = TRUE), dtype = torch_bool(), device = x$device),
       index = idx, B = B, L = L, counts = cnt)
}

#' @rdname to_dense_batch
#' @param dense Result of `to_dense_batch()` with `x` replaced by a new
#'   `B` by `L` by `F'` tensor.
#' @export
from_dense_batch <- function(dense) {
  d <- dense$x
  d$reshape(c(-1L, d$size(3L)))$index_select(1L, dense$index)
}

#' Graph transformer layer with structural attention bias
#'
#' Full multi head self attention among the nodes of each graph (never across
#' graphs of a batch), with a learned scalar bias per head for every
#' shortest path distance, as in Graphormer (Ying et al. 2021). Uses the
#' pre-layer-norm arrangement: \eqn{h = h + \mathrm{MHA}(\mathrm{LN}(h))},
#' then \eqn{h = h + \mathrm{FFN}(\mathrm{LN}(h))}.
#'
#' Graphormer's edge encoding along shortest paths is not implemented. Memory
#' is \eqn{O(B L^2 H)} for `B` graphs of at most `L` nodes and `H` heads, so
#' this layer suits small graphs such as molecules; for large graphs use the
#' sparse [transformer_conv()].
#'
#' The layer is permutation equivariant: distances and attention are
#' functions of node pairs, not of node order.
#'
#' @param dim Feature size.
#' @param heads Attention heads (`dim` must be divisible by `heads`).
#' @param max_dist Largest distinguished distance.
#' @param dropout Dropout probability.
#' @return An `nn_module` called as `layer(x, batch, dist)` where `dist` is a
#'   `B` by `L` by `L` long tensor of clipped distances (padding uses
#'   `max_dist + 2`).
#' @examples
#' if (torch::torch_is_installed()) {
#'   layer <- graph_transformer_layer(8, heads = 2)
#'   layer
#' }
#' @export
graph_transformer_layer <- nn_module(
  "graph_transformer_layer",
  initialize = function(dim, heads = 4L, max_dist = 5L, dropout = 0.1) {
    if (dim %% heads != 0L) stopf("`dim` must be divisible by `heads`.")
    self$heads <- heads
    self$dk <- dim %/% heads
    self$dropout <- dropout
    self$ln1 <- nn_layer_norm(dim)
    self$ln2 <- nn_layer_norm(dim)
    self$qkv <- nn_linear(dim, 3L * dim)
    self$proj <- nn_linear(dim, dim)
    self$ffn <- nn_sequential(nn_linear(dim, 2L * dim), nn_gelu(), nn_linear(2L * dim, dim))
    self$dist_bias <- nn_embedding(max_dist + 3L, heads)
  },
  forward = function(x, batch, dist) {
    dense <- to_dense_batch(self$ln1(x), batch)
    B <- dense$B
    L <- dense$L
    H <- self$heads
    qkv <- self$qkv(dense$x)$view(c(B, L, 3L, H, self$dk))$permute(c(3L, 1L, 4L, 2L, 5L))
    q <- qkv[1L, , , , ]
    k <- qkv[2L, , , , ]
    v <- qkv[3L, , , , ]
    if (B == 1L) {
      q <- q$view(c(1L, H, L, self$dk))
      k <- k$view(c(1L, H, L, self$dk))
      v <- v$view(c(1L, H, L, self$dk))
    }
    scores <- torch_matmul(q, k$transpose(3L, 4L)) / sqrt(self$dk)
    bias <- self$dist_bias(dist + 1L)$permute(c(1L, 4L, 2L, 3L))
    scores <- scores + bias
    key_mask <- dense$mask$view(c(B, 1L, 1L, L))
    scores <- scores$masked_fill(key_mask$logical_not(), -1e9)
    att <- nnf_softmax(scores, dim = -1L)
    self$last_attention <- att$detach()
    att <- nnf_dropout(att, p = self$dropout, training = self$training)
    out <- torch_matmul(att, v)$permute(c(1L, 3L, 2L, 4L))$reshape(c(B, L, H * self$dk))
    dense$x <- self$proj(out)
    h <- x + nnf_dropout(from_dense_batch(dense), p = self$dropout, training = self$training)
    h + nnf_dropout(self$ffn(self$ln2(h)), p = self$dropout, training = self$training)
  }
)

batch_distances <- function(edge_index, batch, max_dist, device) {
  ei <- as_r_array(edge_index)
  b <- as.integer(as_r_array(batch))
  B <- max(b)
  cnt <- tabulate(b, B)
  L <- max(cnt)
  ptr <- c(0L, cumsum(cnt))
  arr <- array(max_dist + 2L, c(B, L, L))
  eb <- if (length(ei)) b[ei[1L, ]] else integer()
  for (k in seq_len(B)) {
    sel <- which(eb == k)
    g <- new_graph_data(edge_index = if (length(sel)) ei[, sel, drop = FALSE] - ptr[k]
                                     else matrix(integer(), 2L, 0L),
                        num_nodes = cnt[k])
    arr[k, seq_len(cnt[k]), seq_len(cnt[k])] <- shortest_path_distances(g, max_dist)
  }
  torch_tensor(arr, dtype = torch_long(), device = device)
}

#' Graphormer style graph transformer
#'
#' Input projection plus centrality encoding (a learned embedding of each
#' node's clipped degree, added to the features), `num_layers`
#' [graph_transformer_layer()]s with shortest path attention bias, a final
#' layer norm, and either a node level linear head or global pooling and an
#' MLP head. Shortest path distances are computed in R for each new batch,
#' which costs \eqn{O(\sum_g n_g (n_g + e_g))}; the distances of the 64 most
#' recent distinct batches are cached in the model, so repeated batches (for
#' example full batch training) reuse them. Edge features are not used by
#' this model; for edge aware attention use [transformer_conv()] with
#' `edge_dim`, which also gives a sparse, neighborhood local alternative to
#' dense attention.
#'
#' The forward signature matches [gnn_model()], so the model works with
#' [train_graph_model()].
#'
#' @param input_dim Node feature size.
#' @param hidden_dim Model width.
#' @param output_dim Output size.
#' @param num_layers Transformer layers.
#' @param heads Attention heads.
#' @param max_dist Largest distinguished shortest path distance.
#' @param max_degree Degrees above this share one embedding.
#' @param dropout Dropout probability.
#' @param pool `NULL` for node level output, or `"mean"`, `"sum"`, `"max"`.
#' @return An `nn_module` of class `graph_transformer`.
#' @references Ying et al. (2021) Do transformers really perform badly for
#'   graph representation? NeurIPS.
#' @examples
#' if (torch::torch_is_installed()) {
#'   m <- graph_transformer_model(input_dim = 34, hidden_dim = 16, output_dim = 2, heads = 2)
#'   m
#' }
#' @export
graph_transformer_model <- nn_module(
  "graph_transformer",
  initialize = function(input_dim, hidden_dim = 64L, output_dim = 1L, num_layers = 2L,
                        heads = 4L, max_dist = 5L, max_degree = 10L, dropout = 0.1,
                        pool = "mean") {
    self$config <- list(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim,
                        num_layers = num_layers, heads = heads, max_dist = max_dist,
                        max_degree = max_degree, dropout = dropout, pool = pool)
    self$input <- nn_linear(input_dim, hidden_dim)
    # Shortest path distances of recently seen batches, so repeated batches
    # (every epoch of full batch training, for example) are not recomputed.
    self$dist_cache <- new.env(parent = emptyenv())
    self$dist_cache$entries <- list()
    self$degree_emb <- nn_embedding(max_degree + 1L, hidden_dim)
    self$layers <- nn_module_list(lapply(seq_len(num_layers), function(i) {
      graph_transformer_layer(hidden_dim, heads, max_dist, dropout)
    }))
    self$ln <- nn_layer_norm(hidden_dim)
    self$head <- if (is.null(pool)) nn_linear(hidden_dim, output_dim)
                 else graph_mlp(c(hidden_dim, hidden_dim, output_dim))
  },
  forward = function(x, edge_index, batch = NULL, num_graphs = NULL, return_embedding = FALSE, ...) {
    n <- x$size(1L)
    if (is.null(batch)) batch <- torch_ones(n, dtype = torch_long(), device = x$device)
    deg <- scatter_count(edge_index[2L, ], n)$to(dtype = torch_long())$clamp(max = self$config$max_degree)
    h <- self$input(x) + self$degree_emb(deg + 1L)
    dist <- self$distances(edge_index, batch, x$device)
    for (layer in seq_along(self$layers)) h <- self$layers[[layer]](h, batch, dist)
    h <- self$ln(h)
    if (return_embedding) return(h)
    if (!is.null(self$config$pool)) {
      h <- switch(self$config$pool,
        mean = global_mean_pool(h, batch, num_graphs),
        sum = global_add_pool(h, batch, num_graphs),
        max = global_max_pool(h, batch, num_graphs))
    }
    self$head(h)
  },
  distances = function(edge_index, batch, device) {
    ei <- as_r_array(edge_index)
    b <- as.integer(as_r_array(batch))
    cache <- self$dist_cache
    for (e in cache$entries) {
      if (length(e$b) == length(b) && length(e$ei) == length(ei) && identical(e$ei, ei) && identical(e$b, b)) {
        return(e$dist$to(device = device))
      }
    }
    dist <- batch_distances(edge_index, batch, self$config$max_dist, device)
    # ponytail: linear scan over at most 64 cached batches; key by a hash if
    # many distinct batches repeat.
    cache$entries <- c(list(list(ei = ei, b = b, dist = dist)), cache$entries)[seq_len(min(64L, length(cache$entries) + 1L))]
    dist
  }
)
