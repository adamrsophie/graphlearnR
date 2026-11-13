#' Chebyshev spectral graph convolution
#'
#' The ChebConv layer of Defferrard, Bresson and Vandergheynst (2016):
#' \deqn{X' = \sum_{k=1}^{K} T_{k-1}(\hat L) X W_k + b,}
#' with Chebyshev polynomials \eqn{T_0(\hat L) X = X},
#' \eqn{T_1(\hat L) X = \hat L X} and
#' \eqn{T_k = 2 \hat L T_{k-1} - T_{k-2}}, and the scaled Laplacian
#' \eqn{\hat L = 2L / \lambda_{max} - I} for the symmetric normalized
#' Laplacian \eqn{L = I - D^{-1/2} A D^{-1/2}}. As in PyTorch Geometric's
#' default, \eqn{\lambda_{max} = 2}, so \eqn{\hat L = -D^{-1/2} A D^{-1/2}}
#' (self loops are removed first). The filter reaches `K - 1` hops. Degrees
#' are weighted in degrees, so the operator is the symmetric Laplacian for
#' undirected graphs.
#'
#' @param in_features,out_features Feature sizes.
#' @param K Number of Chebyshev terms (filter order plus one).
#' @param bias Learn an additive bias.
#' @return An `nn_module` called as `layer(x, edge_index, edge_weight = NULL)`.
#' @references Defferrard, Bresson and Vandergheynst (2016) Convolutional
#'   neural networks on graphs with fast localized spectral filtering.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- dataset_karate()
#'   tt <- graph_tensors(g)
#'   cheb_conv(34, 8, K = 3)(tt$x, tt$edge_index)$shape
#' }
#' @export
cheb_conv <- nn_module(
  "cheb_conv",
  initialize = function(in_features, out_features, K = 3L, bias = TRUE) {
    self$K <- check_count(K, "K", 1L)
    self$lins <- nn_module_list(lapply(seq_len(self$K), function(k) {
      nn_linear(in_features, out_features, bias = FALSE)
    }))
    if (bias) self$bias <- nn_parameter(torch_zeros(out_features))
  },
  forward = function(x, edge_index, edge_weight = NULL, ...) {
    n <- x$size(1L)
    rl <- remove_self_loops_tensor(edge_index, edge_weight)
    nrm <- gcn_norm(rl$edge_index, n, rl$edge_weight, add_self_loops = FALSE, dtype = x$dtype)
    w <- -nrm$edge_weight$to(dtype = x$dtype)
    lap <- function(h) message_passing(h, nrm$edge_index, aggr = "sum", edge_weight = w, num_nodes = n)
    t0 <- x
    out <- self$lins[[1L]](t0)
    if (self$K > 1L) {
      t1 <- lap(x)
      out <- out + self$lins[[2L]](t1)
      for (k in seq_len(self$K - 2L) + 2L) {
        t2 <- 2 * lap(t1) - t0
        out <- out + self$lins[[k]](t2)
        t0 <- t1
        t1 <- t2
      }
    }
    if (!is.null(self$bias)) out <- out + self$bias
    out
  }
)

#' Personalized PageRank propagation (APPNP)
#'
#' `appnp()` propagates features with the approximate personalized PageRank
#' iteration of Klicpera, Bojchevski and Günnemann (2019),
#' \deqn{H^{(0)} = X,\quad H^{(k+1)} = (1 - \alpha) \hat A H^{(k)} + \alpha X,}
#' for `K` steps, with \eqn{\hat A} the GCN propagation matrix (self loops
#' added, symmetric normalization). It has no parameters. As `K` grows the
#' result approaches \eqn{\alpha (I - (1 - \alpha)\hat A)^{-1} X}.
#'
#' `appnp_model()` is the full APPNP model: an MLP predicts per node and the
#' predictions are then propagated. It can be passed to
#' [train_graph_model()] for node tasks.
#'
#' @param K Propagation steps.
#' @param alpha Teleport (restart) probability.
#' @param dropout For `appnp()`, dropout on the propagation weights during
#'   training; for `appnp_model()`, dropout in the MLP.
#' @return An `nn_module`.
#' @references Klicpera, Bojchevski and Günnemann (2019) Predict then
#'   propagate: graph neural networks meet personalized PageRank.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   m <- appnp_model(34, 16, 2, K = 10, alpha = 0.1)
#'   fit <- train_graph_model(m, g, "node_classification", epochs = 50, seed = 1)
#'   evaluate_graph_model(fit, g)
#' }
#' @export
appnp <- nn_module(
  "appnp",
  initialize = function(K = 10L, alpha = 0.1, dropout = 0) {
    self$K <- check_count(K, "K", 1L)
    if (alpha < 0 || alpha > 1) stopf("`alpha` must be in [0, 1].")
    self$alpha <- alpha
    self$dropout <- dropout
  },
  forward = function(x, edge_index, edge_weight = NULL, ...) {
    n <- x$size(1L)
    nrm <- gcn_norm(edge_index, n, edge_weight, dtype = x$dtype)
    # R scalars become float32 tensors; keep the constants in the input's precision.
    keep <- torch_tensor(1 - self$alpha, dtype = x$dtype, device = x$device)
    tele <- torch_tensor(self$alpha, dtype = x$dtype, device = x$device)
    h <- x
    for (k in seq_len(self$K)) {
      w <- nnf_dropout(nrm$edge_weight$to(dtype = x$dtype), p = self$dropout,
                       training = self$training)
      h <- keep * message_passing(h, nrm$edge_index, aggr = "sum", edge_weight = w,
                                  num_nodes = n) + tele * x
    }
    h
  }
)

#' @rdname appnp
#' @param input_dim,hidden_dim,output_dim Feature sizes of the MLP.
#' @export
appnp_model <- nn_module(
  "appnp_model",
  initialize = function(input_dim, hidden_dim = 64L, output_dim = 2L, K = 10L, alpha = 0.1,
                        dropout = 0.5) {
    self$lin1 <- nn_linear(input_dim, hidden_dim)
    self$lin2 <- nn_linear(hidden_dim, output_dim)
    self$prop <- appnp(K, alpha)
    self$dropout <- dropout
    self$config <- list(architecture = "appnp", input_dim = input_dim, hidden_dim = hidden_dim,
                        output_dim = output_dim)
  },
  forward = function(x, edge_index, edge_weight = NULL, return_embedding = FALSE, ...) {
    h <- nnf_dropout(x, p = self$dropout, training = self$training)
    h <- nnf_relu(self$lin1(h))
    if (return_embedding) return(h)
    h <- self$lin2(nnf_dropout(h, p = self$dropout, training = self$training))
    self$prop(h, edge_index, edge_weight)
  }
)

#' Principal neighborhood aggregation (PNA)
#'
#' The PNA layer of Corso, Cavalleri, Beaini, Liò and Veličković (2020), in
#' a single tower form modeled on PyTorch Geometric's `PNAConv` (not
#' numerically compared with it). For node \eqn{i}
#' with in degree \eqn{d_i}:
#'
#' 1. messages \eqn{m_{ji} = U [x_i \Vert x_j \Vert E e_{ji}]} (the edge term
#'    only with `edge_dim`);
#' 2. each aggregator (mean, min, max, std, sum) is applied to the incoming
#'    messages and each result is multiplied by each scaler: identity,
#'    amplification \eqn{\log(d_i + 1) / \delta}, attenuation
#'    \eqn{\delta / \log(d_i + 1)}, linear \eqn{d_i / \bar d} or inverse
#'    linear \eqn{\bar d / d_i}, with \eqn{d_i} clamped to at least 1 and
#'    \eqn{\delta}, \eqn{\bar d} the mean of \eqn{\log(d + 1)} and of
#'    \eqn{d} over the training degree histogram `deg`;
#' 3. \eqn{x_i' = W_2 \, M [x_i \Vert \text{scaled aggregates}]}.
#'
#' The std aggregator is \eqn{\sqrt{\max(\mathrm{var}, 0) + 10^{-5}}}, so
#' nodes without incoming edges, or with
#' identical incoming messages, get \eqn{\sqrt{10^{-5}}} from it and zeros
#' from every other aggregator. Other implementations may treat zero
#' variance differently.
#'
#' `pna_degree()` computes the in degree histogram `deg` from training
#' graphs: element \eqn{d + 1} counts nodes with in degree \eqn{d}.
#'
#' @param in_features,out_features Feature sizes.
#' @param deg Degree histogram from `pna_degree()`.
#' @param aggregators Any of `"mean"`, `"min"`, `"max"`, `"std"`, `"sum"`.
#' @param scalers Any of `"identity"`, `"amplification"`, `"attenuation"`,
#'   `"linear"`, `"inverse_linear"`.
#' @param edge_dim Edge feature size, or `NULL`.
#' @return `pna_conv()`: an `nn_module` called as
#'   `layer(x, edge_index, edge_attr = NULL)`. `pna_degree()`: an integer
#'   vector.
#' @references Corso, Cavalleri, Beaini, Liò and Veličković (2020) Principal
#'   neighbourhood aggregation for graph nets.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- dataset_karate()
#'   tt <- graph_tensors(g)
#'   layer <- pna_conv(34, 8, deg = pna_degree(g))
#'   layer(tt$x, tt$edge_index)$shape
#' }
#' @export
pna_conv <- nn_module(
  "pna_conv",
  initialize = function(in_features, out_features, deg,
                        aggregators = c("mean", "min", "max", "std"),
                        scalers = c("identity", "amplification", "attenuation"),
                        edge_dim = NULL) {
    ok_a <- c("mean", "min", "max", "std", "sum")
    ok_s <- c("identity", "amplification", "attenuation", "linear", "inverse_linear")
    if (!length(aggregators) || !all(aggregators %in% ok_a)) {
      stopf("`aggregators` must be among %s.", paste(ok_a, collapse = ", "))
    }
    if (!length(scalers) || !all(scalers %in% ok_s)) {
      stopf("`scalers` must be among %s.", paste(ok_s, collapse = ", "))
    }
    deg <- as.numeric(deg)
    if (!length(deg) || sum(deg) <= 0) stopf("`deg` must be a degree histogram from pna_degree().")
    bins <- seq_along(deg) - 1
    if (sum(bins * deg) <= 0) {
      stopf("`deg` counts only nodes of degree zero; the PNA scalers need graphs with edges.")
    }
    self$avg_lin <- sum(bins * deg) / sum(deg)
    self$avg_log <- sum(log(bins + 1) * deg) / sum(deg)
    self$aggregators <- aggregators
    self$scalers <- scalers
    n_in <- 2L
    if (!is.null(edge_dim)) {
      self$edge_encoder <- nn_linear(edge_dim, in_features)
      n_in <- 3L
    }
    self$pre <- nn_linear(n_in * in_features, in_features)
    self$post <- nn_linear((length(aggregators) * length(scalers) + 1L) * in_features, out_features)
    self$lin <- nn_linear(out_features, out_features)
  },
  forward = function(x, edge_index, edge_attr = NULL, ...) {
    n <- x$size(1L)
    src <- edge_index[1L, ]
    dst <- edge_index[2L, ]
    parts <- list(x$index_select(1L, dst), x$index_select(1L, src))
    if (!is.null(self$edge_encoder)) {
      if (is.null(edge_attr)) stopf("This PNA layer needs `edge_attr`.")
      parts[[3L]] <- self$edge_encoder(edge_attr)
    }
    m <- self$pre(torch_cat(parts, dim = 2L))
    agg <- torch_cat(lapply(self$aggregators, function(a) scatter_aggregate(m, dst, n, a)), dim = 2L)
    d <- scatter_count(dst, n, dtype = x$dtype, device = x$device)$clamp(min = 1)$unsqueeze(2L)
    scaled <- lapply(self$scalers, function(s) {
      switch(s,
        identity = agg,
        amplification = agg * (torch_log(d + 1) / self$avg_log),
        attenuation = agg * (self$avg_log / torch_log(d + 1)),
        linear = agg * (d / self$avg_lin),
        inverse_linear = agg * (self$avg_lin / d)
      )
    })
    self$lin(self$post(torch_cat(c(list(x), scaled), dim = 2L)))
  }
)

#' @rdname pna_conv
#' @param graphs A `graph_data` object or a list of them.
#' @export
pna_degree <- function(graphs) {
  if (inherits(graphs, "graph_data")) graphs <- list(graphs)
  d <- unlist(lapply(graphs, function(g) tabulate(g$edge_index[2L, ], nbins = g$num_nodes)))
  tabulate(d + 1L, nbins = max(d) + 1L)
}
