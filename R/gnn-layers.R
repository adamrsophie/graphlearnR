#' Graph neural network layers
#'
#' Layers are `torch` modules built on [message_passing()]. Every layer takes
#' node features `x` (an N by F tensor, or `list(x_src, x_dst)` for bipartite
#' relations where noted) and a one based long `edge_index` (2 by E, row 1
#' source, row 2 target), and returns one row per target node. Messages flow
#' from source to target, so a node aggregates over its **incoming** edges.
#' For undirected graphs, [graph_data()] stores both directions, so this is
#' the usual neighborhood.
#'
#' | Layer | Update for node \eqn{i} | Edge input | Self loops |
#' |---|---|---|---|
#' | `sage_conv` | \eqn{W_1 \mathrm{aggr}_{j} w_{ji} x_j + W_2 x_i} | `edge_weight` | none added |
#' | `gcn_conv` | \eqn{\sum_j \frac{w_{ji}}{\sqrt{d_j d_i}} W x_j + b} | `edge_weight` | added to nodes without one |
#' | `graph_conv` | \eqn{W_1 \sum_j w_{ji} x_j + W_2 x_i} | `edge_weight` | none added |
#' | `gin_conv` | \eqn{\mathrm{MLP}((1+\epsilon) x_i + \sum_j x_j)} | `edge_attr` (GINE), `edge_weight` | none added |
#' | `gat_conv` | \eqn{\sum_j \alpha_{ij} W x_j}, \eqn{\alpha} from \eqn{a^\top [W x_i \Vert W x_j]} | `edge_attr` | replaced: existing removed, one per node added |
#' | `gatv2_conv` | \eqn{\sum_j \alpha_{ij} W_s x_j}, \eqn{\alpha} from \eqn{a^\top \mathrm{LeakyReLU}(W_s x_j + W_t x_i)} | `edge_attr` | as `gat_conv` |
#' | `transformer_conv` | \eqn{W_r x_i + \sum_j \alpha_{ij} (W_v x_j + W_e e_{ji})} | `edge_attr` | none added |
#' | `nn_conv` | \eqn{W x_i + \mathrm{aggr}_j x_j \Theta(e_{ji})} | `edge_attr` (required) | none added |
#' | `rgcn_conv` | \eqn{W_0 x_i + \sum_r \frac{1}{|N_r(i)|}\sum_{j \in N_r(i)} W_r x_j} | `edge_type` (required) | none added |
#'
#' Cost is \eqn{O(E F)} for message construction plus the dense transforms
#' (\eqn{O(N F F')}); attention layers keep an E by heads attention tensor;
#' `nn_conv` stores an E by F by F' tensor of edge specific weights and is the
#' most memory hungry. All layers are permutation equivariant: permuting node
#' rows and relabeling `edge_index` permutes the output rows the same way.
#'
#' The formulas follow the PyTorch Geometric layers of the same names
#' (`SAGEConv`, `GCNConv`, `GraphConv`, `GINConv`/`GINEConv`, `GATConv`,
#' `GATv2Conv`, `TransformerConv`, `NNConv`, `RGCNConv`) for the options
#' exposed here; options not listed are not implemented.
#'
#' @param in_features Input feature size. `sage_conv`, `graph_conv`,
#'   `gat_conv`, `gatv2_conv` and `transformer_conv` also accept
#'   `c(source, target)` sizes for bipartite input.
#' @param out_features Output feature size (per head for attention layers).
#' @return An `nn_module`.
#' @references
#' Hamilton, Ying and Leskovec (2017) Inductive representation learning on
#' large graphs. Kipf and Welling (2017) Semi-supervised classification with
#' graph convolutional networks. Xu et al. (2019) How powerful are graph
#' neural networks? Velickovic et al. (2018) Graph attention networks. Brody,
#' Alon and Yahav (2022) How attentive are graph attention networks? Shi et
#' al. (2021) Masked label prediction. Gilmer et al. (2017) Neural message
#' passing for quantum chemistry. Schlichtkrull et al. (2018) Modeling
#' relational data with graph convolutional networks. Morris et al. (2019)
#' Weisfeiler and Leman go neural.
#' @examples
#' if (torch::torch_is_installed()) {
#'   x <- torch::torch_randn(4, 3)
#'   ei <- torch::torch_tensor(rbind(c(1L, 2L, 3L), c(2L, 3L, 4L)),
#'                             dtype = torch::torch_long())
#'   layer <- sage_conv(3, 5)
#'   layer(x, ei)$shape
#'   gat_conv(3, 2, heads = 2)(x, ei)$shape
#' }
#' @name gnn-layers
NULL

is_pair <- function(x) is.list(x) && !inherits(x, "torch_tensor")
dst_of <- function(x) if (is_pair(x)) x[[2L]] else x
src_of <- function(x) if (is_pair(x)) x[[1L]] else x
in_src <- function(in_features) in_features[1L]
in_dst <- function(in_features) in_features[length(in_features)]

glorot_param <- function(...) {
  p <- torch_empty(...)
  nn_init_xavier_uniform_(p)
  nn_parameter(p)
}

#' @rdname gnn-layers
#' @param aggr Aggregation, see [scatter_aggregate()].
#' @param root_weight Add a transformed copy of the target node's own
#'   features.
#' @param normalize For `sage_conv`, L2 normalize the output rows; for
#'   `gcn_conv`, apply the symmetric degree normalization.
#' @param bias Learn an additive bias.
#' @export
sage_conv <- nn_module(
  "sage_conv",
  initialize = function(in_features, out_features, aggr = "mean",
                        root_weight = TRUE, normalize = FALSE, bias = TRUE) {
    self$aggr <- aggr
    self$root_weight <- root_weight
    self$normalize <- normalize
    self$lin_l <- nn_linear(in_src(in_features) * length(aggr), out_features, bias = bias)
    if (root_weight) {
      self$lin_r <- nn_linear(in_dst(in_features), out_features, bias = FALSE)
    }
  },
  forward = function(x, edge_index, edge_weight = NULL, ...) {
    h <- message_passing(x, edge_index, aggr = self$aggr, edge_weight = edge_weight)
    out <- self$lin_l(h)
    if (self$root_weight) out <- out + self$lin_r(dst_of(x))
    if (self$normalize) out <- nnf_normalize(out, p = 2, dim = -1L)
    out
  }
)

#' @rdname gnn-layers
#' @param add_self_loops Add self loops before propagation (never for a pair
#'   of source and target feature sets, whose index spaces differ).
#' @param improved Self loop weight 2 instead of 1 (`gcn_conv`).
#' @export
gcn_conv <- nn_module(
  "gcn_conv",
  initialize = function(in_features, out_features, add_self_loops = TRUE,
                        normalize = TRUE, improved = FALSE, bias = TRUE) {
    self$add_loops <- add_self_loops
    self$normalize <- normalize
    self$improved <- improved
    self$lin <- nn_linear(in_features, out_features, bias = FALSE)
    nn_init_xavier_uniform_(self$lin$weight)
    if (bias) self$bias <- nn_parameter(torch_zeros(out_features))
  },
  forward = function(x, edge_index, edge_weight = NULL, ...) {
    n <- x$size(1L)
    h <- self$lin(x)
    if (self$normalize) {
      nrm <- gcn_norm(edge_index, n, edge_weight, self$add_loops, self$improved)
      edge_index <- nrm$edge_index
      edge_weight <- nrm$edge_weight
    } else if (self$add_loops) {
      w <- edge_weight %||% torch_ones(edge_index$size(2L), device = x$device)
      sl <- add_self_loops_tensor(edge_index, n, w,
                                  fill_value = if (self$improved) 2 else 1,
                                  remaining = TRUE)
      edge_index <- sl$edge_index
      edge_weight <- sl$edge_weight
    }
    out <- message_passing(h, edge_index, aggr = "sum", edge_weight = edge_weight)
    if (!is.null(self$bias)) out <- out + self$bias
    out
  }
)

#' @rdname gnn-layers
#' @export
graph_conv <- nn_module(
  "graph_conv",
  initialize = function(in_features, out_features, aggr = "sum", bias = TRUE) {
    self$aggr <- aggr
    self$lin_rel <- nn_linear(in_src(in_features) * length(aggr), out_features, bias = bias)
    self$lin_root <- nn_linear(in_dst(in_features), out_features, bias = FALSE)
  },
  forward = function(x, edge_index, edge_weight = NULL, ...) {
    h <- message_passing(x, edge_index, aggr = self$aggr, edge_weight = edge_weight)
    self$lin_rel(h) + self$lin_root(dst_of(x))
  }
)

#' @rdname gnn-layers
#' @param nn A module applied after aggregation (`gin_conv`), for example an
#'   MLP from [graph_mlp()]. For `gin_conv` its input size must equal the
#'   node feature size.
#' @param eps Initial \eqn{\epsilon}.
#' @param train_eps Learn \eqn{\epsilon}.
#' @param edge_dim Edge feature size. For `gin_conv` this turns on the GINE
#'   variant, \eqn{\sum_j \mathrm{ReLU}(x_j + W_e e_{ji})}; for attention layers
#'   edge features enter the attention scores.
#' @export
gin_conv <- nn_module(
  "gin_conv",
  initialize = function(nn, eps = 0, train_eps = FALSE, edge_dim = NULL,
                        in_features = NULL) {
    self$nn <- nn
    self$train_eps <- train_eps
    if (train_eps) {
      self$eps <- nn_parameter(torch_tensor(eps))
    } else {
      self$eps_value <- eps
    }
    if (!is.null(edge_dim)) {
      if (is.null(in_features)) {
        stopf("`gin_conv()` with `edge_dim` needs `in_features`.")
      }
      self$lin_edge <- nn_linear(edge_dim, in_features)
    }
  },
  forward = function(x, edge_index, edge_attr = NULL, edge_weight = NULL, ...) {
    msg_fun <- NULL
    if (!is.null(self$lin_edge)) {
      if (is.null(edge_attr)) stopf("This GINE layer needs `edge_attr`.")
      lin_edge <- self$lin_edge
      msg_fun <- function(x_j, edge_attr, edge_weight, ...) {
        m <- nnf_relu(x_j + lin_edge(edge_attr))
        if (is.null(edge_weight)) m else m * edge_weight$unsqueeze(-1L)
      }
    }
    h <- message_passing(x, edge_index, message = msg_fun, aggr = "sum",
                         edge_attr = edge_attr, edge_weight = edge_weight)
    eps <- if (self$train_eps) self$eps else self$eps_value
    self$nn((1 + eps) * dst_of(x) + h)
  }
)

#' @rdname gnn-layers
#' @param heads Number of attention heads.
#' @param concat Concatenate heads (output size `heads * out_features`);
#'   otherwise average them.
#' @param negative_slope Slope of the LeakyReLU in the attention score.
#' @param dropout Dropout probability on attention coefficients (training
#'   only).
#' @details Attention layers keep the coefficients of their last forward pass
#'   in `layer$last_attention` (a list with `edge_index` and an E by heads
#'   tensor `alpha`). Attention coefficients describe how the model weighted
#'   neighbors; they are not by themselves measures of feature importance or
#'   causal relevance.
#' @export
gat_conv <- nn_module(
  "gat_conv",
  initialize = function(in_features, out_features, heads = 1L, concat = TRUE,
                        negative_slope = 0.2, dropout = 0,
                        add_self_loops = TRUE, edge_dim = NULL, bias = TRUE) {
    self$heads <- heads
    self$out_features <- out_features
    self$concat <- concat
    self$negative_slope <- negative_slope
    self$dropout <- dropout
    self$add_loops <- add_self_loops
    self$bipartite <- length(in_features) > 1L
    self$lin_src <- nn_linear(in_src(in_features), heads * out_features, bias = FALSE)
    nn_init_xavier_uniform_(self$lin_src$weight)
    if (self$bipartite) {
      self$lin_dst <- nn_linear(in_dst(in_features), heads * out_features, bias = FALSE)
      nn_init_xavier_uniform_(self$lin_dst$weight)
    }
    self$att_src <- glorot_param(1L, heads, out_features)
    self$att_dst <- glorot_param(1L, heads, out_features)
    if (!is.null(edge_dim)) {
      self$lin_edge <- nn_linear(edge_dim, heads * out_features, bias = FALSE)
      self$att_edge <- glorot_param(1L, heads, out_features)
    }
    if (bias) {
      self$bias <- nn_parameter(torch_zeros(if (concat) heads * out_features else out_features))
    }
  },
  forward = function(x, edge_index, edge_attr = NULL, ...) {
    H <- self$heads
    C <- self$out_features
    pair <- is.list(x)
    xs <- self$lin_src(src_of(x))$view(c(-1L, H, C))
    # A pair of feature sets with one input size shares the source map.
    xd <- if (self$bipartite) self$lin_dst(dst_of(x))$view(c(-1L, H, C))
          else if (pair) self$lin_src(dst_of(x))$view(c(-1L, H, C)) else xs
    n_dst <- xd$size(1L)
    if (self$add_loops && !self$bipartite && !pair) {
      rl <- remove_self_loops_tensor(edge_index, edge_attr = edge_attr)
      sl <- add_self_loops_tensor(rl$edge_index, n_dst, edge_attr = rl$edge_attr)
      edge_index <- sl$edge_index
      edge_attr <- sl$edge_attr
    }
    src <- edge_index[1L, ]
    dst <- edge_index[2L, ]
    a_s <- (xs * self$att_src)$sum(dim = -1L)
    a_d <- (xd * self$att_dst)$sum(dim = -1L)
    alpha <- a_s$index_select(1L, src) + a_d$index_select(1L, dst)
    if (!is.null(self$lin_edge)) {
      if (is.null(edge_attr)) stopf("This attention layer needs `edge_attr`.")
      e <- self$lin_edge(edge_attr)$view(c(-1L, H, C))
      alpha <- alpha + (e * self$att_edge)$sum(dim = -1L)
    }
    alpha <- nnf_leaky_relu(alpha, self$negative_slope)
    alpha <- scatter_softmax(alpha, dst, n_dst)
    self$last_attention <- list(edge_index = edge_index, alpha = alpha$detach())
    alpha <- nnf_dropout(alpha, p = self$dropout, training = self$training)
    msg <- xs$index_select(1L, src) * alpha$unsqueeze(-1L)
    out <- scatter_sum(msg$reshape(c(-1L, H * C)), dst, n_dst)$view(c(-1L, H, C))
    out <- if (self$concat) out$reshape(c(-1L, H * C)) else out$mean(dim = 2L)
    if (!is.null(self$bias)) out <- out + self$bias
    out
  }
)

#' @rdname gnn-layers
#' @export
gatv2_conv <- nn_module(
  "gatv2_conv",
  initialize = function(in_features, out_features, heads = 1L, concat = TRUE,
                        negative_slope = 0.2, dropout = 0,
                        add_self_loops = TRUE, edge_dim = NULL, bias = TRUE) {
    self$heads <- heads
    self$out_features <- out_features
    self$concat <- concat
    self$negative_slope <- negative_slope
    self$dropout <- dropout
    self$add_loops <- add_self_loops
    self$bipartite <- length(in_features) > 1L
    self$lin_l <- nn_linear(in_src(in_features), heads * out_features, bias = bias)
    self$lin_r <- nn_linear(in_dst(in_features), heads * out_features, bias = bias)
    self$att <- glorot_param(1L, heads, out_features)
    if (!is.null(edge_dim)) {
      self$lin_edge <- nn_linear(edge_dim, heads * out_features, bias = FALSE)
    }
    if (bias) {
      self$bias <- nn_parameter(torch_zeros(if (concat) heads * out_features else out_features))
    }
  },
  forward = function(x, edge_index, edge_attr = NULL, ...) {
    H <- self$heads
    C <- self$out_features
    xl <- self$lin_l(src_of(x))$view(c(-1L, H, C))
    xr <- self$lin_r(dst_of(x))$view(c(-1L, H, C))
    n_dst <- xr$size(1L)
    if (self$add_loops && !self$bipartite && !is.list(x)) {
      rl <- remove_self_loops_tensor(edge_index, edge_attr = edge_attr)
      sl <- add_self_loops_tensor(rl$edge_index, n_dst, edge_attr = rl$edge_attr)
      edge_index <- sl$edge_index
      edge_attr <- sl$edge_attr
    }
    src <- edge_index[1L, ]
    dst <- edge_index[2L, ]
    xj <- xl$index_select(1L, src)
    z <- xj + xr$index_select(1L, dst)
    if (!is.null(self$lin_edge)) {
      if (is.null(edge_attr)) stopf("This attention layer needs `edge_attr`.")
      z <- z + self$lin_edge(edge_attr)$view(c(-1L, H, C))
    }
    alpha <- (nnf_leaky_relu(z, self$negative_slope) * self$att)$sum(dim = -1L)
    alpha <- scatter_softmax(alpha, dst, n_dst)
    self$last_attention <- list(edge_index = edge_index, alpha = alpha$detach())
    alpha <- nnf_dropout(alpha, p = self$dropout, training = self$training)
    out <- scatter_sum((xj * alpha$unsqueeze(-1L))$reshape(c(-1L, H * C)), dst, n_dst)
    out <- out$view(c(-1L, H, C))
    out <- if (self$concat) out$reshape(c(-1L, H * C)) else out$mean(dim = 2L)
    if (!is.null(self$bias)) out <- out + self$bias
    out
  }
)

#' @rdname gnn-layers
#' @export
transformer_conv <- nn_module(
  "transformer_conv",
  initialize = function(in_features, out_features, heads = 1L, concat = TRUE,
                        dropout = 0, edge_dim = NULL, root_weight = TRUE,
                        bias = TRUE) {
    self$heads <- heads
    self$out_features <- out_features
    self$concat <- concat
    self$dropout <- dropout
    self$root_weight <- root_weight
    self$lin_query <- nn_linear(in_dst(in_features), heads * out_features, bias = bias)
    self$lin_key <- nn_linear(in_src(in_features), heads * out_features, bias = bias)
    self$lin_value <- nn_linear(in_src(in_features), heads * out_features, bias = bias)
    if (!is.null(edge_dim)) {
      self$lin_edge <- nn_linear(edge_dim, heads * out_features, bias = FALSE)
    }
    if (root_weight) {
      self$lin_skip <- nn_linear(in_dst(in_features),
                                 if (concat) heads * out_features else out_features,
                                 bias = bias)
    }
  },
  forward = function(x, edge_index, edge_attr = NULL, ...) {
    H <- self$heads
    C <- self$out_features
    xd <- dst_of(x)
    q <- self$lin_query(xd)$view(c(-1L, H, C))
    k <- self$lin_key(src_of(x))$view(c(-1L, H, C))
    v <- self$lin_value(src_of(x))$view(c(-1L, H, C))
    n_dst <- q$size(1L)
    src <- edge_index[1L, ]
    dst <- edge_index[2L, ]
    kj <- k$index_select(1L, src)
    vj <- v$index_select(1L, src)
    if (!is.null(self$lin_edge)) {
      if (is.null(edge_attr)) stopf("This attention layer needs `edge_attr`.")
      e <- self$lin_edge(edge_attr)$view(c(-1L, H, C))
      kj <- kj + e
      vj <- vj + e
    }
    alpha <- (q$index_select(1L, dst) * kj)$sum(dim = -1L) / sqrt(C)
    alpha <- scatter_softmax(alpha, dst, n_dst)
    self$last_attention <- list(edge_index = edge_index, alpha = alpha$detach())
    alpha <- nnf_dropout(alpha, p = self$dropout, training = self$training)
    out <- scatter_sum((vj * alpha$unsqueeze(-1L))$reshape(c(-1L, H * C)), dst, n_dst)
    out <- out$view(c(-1L, H, C))
    out <- if (self$concat) out$reshape(c(-1L, H * C)) else out$mean(dim = 2L)
    if (self$root_weight) out <- out + self$lin_skip(xd)
    out
  }
)

#' @rdname gnn-layers
#' @param edge_nn Module mapping edge features (E by `edge_dim`) to
#'   E by `in_features * out_features` (`nn_conv`). When `NULL`, a two layer
#'   MLP with 32 hidden units is built from `edge_dim`.
#' @export
nn_conv <- nn_module(
  "nn_conv",
  initialize = function(in_features, out_features, edge_dim = NULL,
                        edge_nn = NULL, aggr = "sum", root_weight = TRUE,
                        bias = TRUE) {
    if (is.null(edge_nn)) {
      if (is.null(edge_dim)) stopf("`nn_conv()` needs `edge_nn` or `edge_dim`.")
      edge_nn <- nn_sequential(nn_linear(edge_dim, 32L), nn_relu(),
                               nn_linear(32L, in_features * out_features))
    }
    self$edge_nn <- edge_nn
    self$in_features <- in_features
    self$out_features <- out_features
    self$aggr <- aggr
    self$root_weight <- root_weight
    if (root_weight) self$lin_root <- nn_linear(in_features, out_features, bias = FALSE)
    if (bias) self$bias <- nn_parameter(torch_zeros(out_features))
  },
  forward = function(x, edge_index, edge_attr = NULL, ...) {
    if (is.null(edge_attr)) stopf("`nn_conv` needs `edge_attr`.")
    w <- self$edge_nn(edge_attr)$view(c(-1L, self$in_features, self$out_features))
    msg_fun <- function(x_j, ...) torch_bmm(x_j$unsqueeze(2L), w)$squeeze(2L)
    out <- message_passing(x, edge_index, message = msg_fun, aggr = self$aggr)
    if (self$root_weight) out <- out + self$lin_root(x)
    if (!is.null(self$bias)) out <- out + self$bias
    out
  }
)

#' @rdname gnn-layers
#' @param num_relations Number of relation types; `edge_type` holds codes
#'   `1:num_relations`.
#' @param num_bases Use basis decomposition with this many bases (`NULL`
#'   for a full weight per relation).
#' @export
rgcn_conv <- nn_module(
  "rgcn_conv",
  initialize = function(in_features, out_features, num_relations,
                        num_bases = NULL, aggr = "mean", root_weight = TRUE,
                        bias = TRUE) {
    self$num_relations <- num_relations
    self$num_bases <- num_bases
    self$aggr <- aggr
    if (is.null(num_bases)) {
      self$weight <- glorot_param(num_relations, in_features, out_features)
    } else {
      self$basis <- glorot_param(num_bases, in_features, out_features)
      self$comp <- glorot_param(num_relations, num_bases)
    }
    self$root_weight <- root_weight
    if (root_weight) self$lin_root <- nn_linear(in_features, out_features, bias = FALSE)
    if (bias) self$bias <- nn_parameter(torch_zeros(out_features))
  },
  relation_weights = function() {
    if (is.null(self$num_bases)) {
      return(self$weight)
    }
    torch_einsum("rb,bio->rio", list(self$comp, self$basis))
  },
  forward = function(x, edge_index, edge_type, ...) {
    if (missing(edge_type) || is.null(edge_type)) stopf("`rgcn_conv` needs `edge_type`.")
    edge_type <- as_index_tensor(edge_type, x$device)
    if (edge_type$numel() != edge_index$size(2L)) {
      stopf("`edge_type` has %d entries for %d edges.", edge_type$numel(), edge_index$size(2L))
    }
    if (edge_type$numel() > 0L) {
      rng <- as.numeric(as_r_array(torch_stack(list(edge_type$min(), edge_type$max()))))
      if (rng[1L] < 1 || rng[2L] > self$num_relations) {
        stopf("`edge_type` codes must be in 1:%d; found %g to %g.", self$num_relations, rng[1L], rng[2L])
      }
    }
    w <- self$relation_weights()
    n <- x$size(1L)
    out <- torch_zeros(n, w$size(3L), dtype = x$dtype, device = x$device)
    for (r in seq_len(self$num_relations)) {
      keep <- edge_type == r
      if (!as.logical(keep$any()$cpu())) next
      h <- message_passing(x, edge_index[, keep], aggr = self$aggr, num_nodes = n)
      out <- out + torch_matmul(h, w[r, , ])
    }
    if (self$root_weight) out <- out + self$lin_root(x)
    if (!is.null(self$bias)) out <- out + self$bias
    out
  }
)

#' Multilayer perceptron
#'
#' A plain MLP used inside [gin_conv()] and as prediction heads.
#'
#' @param dims Integer vector of layer sizes, including input and output.
#' @param activation Activation module constructor between layers.
#' @param dropout Dropout probability between layers.
#' @param batch_norm Insert batch normalization between layers.
#' @return An `nn_module`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   m <- graph_mlp(c(4, 8, 2))
#'   m(torch::torch_randn(3, 4))$shape
#' }
#' @export
graph_mlp <- function(dims, activation = nn_relu, dropout = 0, batch_norm = FALSE) {
  if (length(dims) < 2L) stopf("`dims` needs at least an input and an output size.")
  layers <- list()
  k <- length(dims) - 1L
  for (i in seq_len(k)) {
    layers[[length(layers) + 1L]] <- nn_linear(dims[i], dims[i + 1L])
    if (i < k) {
      if (batch_norm) layers[[length(layers) + 1L]] <- nn_batch_norm1d(dims[i + 1L])
      layers[[length(layers) + 1L]] <- activation()
      if (dropout > 0) layers[[length(layers) + 1L]] <- nn_dropout(dropout)
    }
  }
  do.call(nn_sequential, layers)
}
