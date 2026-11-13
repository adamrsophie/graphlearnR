#' Diffusion on a graph, solved with modelcompR
#'
#' Solves the graph heat equation
#' \deqn{\frac{du}{dt} = -\kappa\, L\, u,\quad u(0) = u_0,}
#' with \eqn{L = D - A} the Laplacian (`"laplacian"`) or
#' \eqn{I - D^{-1/2} A D^{-1/2}} (`"normalized_laplacian"`), where
#' \eqn{A_{ij}} is the weight of the edge from \eqn{j} to \eqn{i} and
#' \eqn{D} holds the weighted in degrees. For an undirected graph the
#' solution conserves \eqn{\sum_i u_i} under the plain Laplacian and tends to
#' the mean.
#'
#' The ODE is solved by [modelcompR](https://github.com/adamrsophie/modelcompR)
#' (`ODEProblem()` and `solve()` with `Tsit5()` by default); graphlearnR has
#' no integrator of its own. This is an optional bridge: modelcompR must be
#' installed, and the solve runs on R numerics, so no gradients flow back to
#' torch. It is a mechanistic simulation tool, not a neural ODE layer.
#'
#' @param g A `graph_data` object.
#' @param u0 Initial value per node (numeric vector of length `num_nodes`).
#' @param times Times at which to return the solution; the first is the start.
#' @param rate Diffusion rate \eqn{\kappa}.
#' @param operator `"laplacian"` or `"normalized_laplacian"`.
#' @param alg A modelcompR algorithm; default `modelcompR::Tsit5()`.
#' @param ... Passed to modelcompR's `solve()` (for example `abstol`).
#' @return A matrix with one row per time and one column per node, with the
#'   times in attribute `"times"`.
#' @examples
#' if (requireNamespace("modelcompR", quietly = TRUE)) {
#'   g <- graph_data(edge_index = rbind(1:3, 2:4), num_nodes = 4, directed = FALSE)
#'   graph_diffusion(g, c(1, 0, 0, 0), times = c(0, 1, 5))
#' }
#' @export
graph_diffusion <- function(g, u0, times, rate = 1,
                            operator = c("laplacian", "normalized_laplacian"),
                            alg = NULL, ...) {
  if (!requireNamespace("modelcompR", quietly = TRUE)) {
    stopf("graph_diffusion() needs the modelcompR package: remotes::install_github(\"adamrsophie/modelcompR\").")
  }
  operator <- match.arg(operator)
  if (!inherits(g, "graph_data")) stopf("`g` must be a graph_data object.")
  if (length(u0) != g$num_nodes) stopf("`u0` must have one value per node (%d).", g$num_nodes)
  times <- as.numeric(times)
  if (length(times) < 2L || is.unsorted(times, strictly = TRUE)) {
    stopf("`times` must be increasing with at least two values.")
  }
  L <- graph_laplacian(g, operator)
  f <- function(u, p, t) as.vector(-p$rate * (p$L %*% u))
  prob <- modelcompR::ODEProblem(f, as.numeric(u0), range(times), list(L = L, rate = rate))
  sol <- solve(prob, alg %||% modelcompR::Tsit5(), saveat = times, ...)
  if (!identical(as.character(sol$retcode), "Success")) {
    stopf("modelcompR did not solve the problem (retcode %s).", as.character(sol$retcode))
  }
  out <- do.call(rbind, lapply(sol$u, as.numeric))
  dimnames(out) <- list(NULL, g$node_ids %||% NULL)
  attr(out, "times") <- sol$t
  out
}

graph_laplacian <- function(g, operator) {
  A <- Matrix::t(adjacency_from_graph(g, sparse = TRUE, weighted = TRUE))
  d <- Matrix::rowSums(A)
  if (operator == "laplacian") return(Matrix::Diagonal(x = d) - A)
  s <- ifelse(d > 0, 1 / sqrt(d), 0)
  Matrix::Diagonal(length(d)) - Matrix::Diagonal(x = s) %*% A %*% Matrix::Diagonal(x = s)
}

#' Exact graph operators with symcompR
#'
#' Returns a graph operator as an exact symbolic matrix from
#' [symcompR](https://github.com/adamrsophie/symcompR): the adjacency
#' matrix, the Laplacian \eqn{D - A}, or the GCN propagation matrix
#' \eqn{\hat D^{-1/2} (A + I) \hat D^{-1/2}} used by [gcn_conv()], with
#' square roots kept exact. Row \eqn{i}, column \eqn{j} is the weight from
#' \eqn{j} to \eqn{i}. Edge weights are ignored (unweighted operators), so
#' every entry is an integer, a rational or an algebraic number.
#'
#' This is an optional bridge, useful for checking layer normalizations
#' against exact values or showing the operator a layer applies. It needs
#' symcompR and is meant for small graphs.
#'
#' @param g A `graph_data` object.
#' @param operator `"adjacency"`, `"laplacian"` or `"gcn"`.
#' @return A symcompR matrix; `as.matrix()` gives its numeric value.
#' @examples
#' if (requireNamespace("symcompR", quietly = TRUE)) {
#'   g <- graph_data(edge_index = rbind(1:2, 2:3), num_nodes = 3, directed = FALSE)
#'   graph_operator_symbolic(g, "gcn")
#' }
#' @export
graph_operator_symbolic <- function(g, operator = c("adjacency", "laplacian", "gcn")) {
  if (!requireNamespace("symcompR", quietly = TRUE)) {
    stopf("graph_operator_symbolic() needs the symcompR package: remotes::install_github(\"adamrsophie/symcompR\").")
  }
  operator <- match.arg(operator)
  if (!inherits(g, "graph_data")) stopf("`g` must be a graph_data object.")
  A <- t(as.matrix(adjacency_from_graph(g, sparse = FALSE, weighted = FALSE)))
  n <- nrow(A)
  int <- symcompR::Integer
  entry <- switch(operator,
    adjacency = function(i, j) int(A[i, j]),
    laplacian = {
      d <- rowSums(A)
      function(i, j) int((i == j) * d[i] - A[i, j])
    },
    gcn = {
      Ah <- A
      diag(Ah)[diag(Ah) == 0] <- 1
      d <- rowSums(Ah)
      function(i, j) if (Ah[i, j] == 0) int(0) else int(Ah[i, j]) / sqrt(int(d[i] * d[j]))
    }
  )
  symcompR::Matrix(lapply(seq_len(n), function(i) lapply(seq_len(n), function(j) entry(i, j))))
}

#' Rule constrained node labels with z3r
#'
#' Finds the most probable joint labeling of the nodes that satisfies
#' logical rules on edges, using the Z3 optimizer through
#' [z3r](https://github.com/adamrsophie/z3r). Each node \eqn{i} gets one
#' label \eqn{c_i}; the objective is \eqn{\max \sum_i \log p_i(c_i)} (log
#' probabilities rounded to `precision`) subject to: no edge \eqn{(j, i)}
#' has \eqn{(c_j, c_i)} in `forbidden`, and nodes in `fixed` keep their
#' label. This turns independent per node predictions into a consistent
#' assignment, for example when two adjacent nodes cannot both belong to a
#' class.
#'
#' This is an optional bridge for small graphs: the problem is a weighted
#' constraint satisfaction problem with one integer variable per node, and
#' solving time grows quickly with the number of nodes and rules. It runs
#' after training and is not differentiable.
#'
#' @param probs Matrix of class probabilities, one row per node and one
#'   named column per class (for example from the `.prob_` columns of
#'   [predict()]).
#' @param edge_index Integer matrix, 2 by E, 1-based.
#' @param forbidden Two column matrix or data frame of class names: an edge
#'   from a node of the first class to a node of the second is not allowed.
#' @param fixed Optional named vector of known labels; names are node
#'   indices.
#' @param precision Log probabilities are multiplied by this and rounded.
#' @param timeout Solver timeout in milliseconds.
#' @return A list with `labels` (factor), `status` (`"sat"`, `"unsat"` or
#'   `"unknown"`), `changed` (nodes whose label differs from the most
#'   probable class) and `log_prob` (total log probability of the result).
#' @examples
#' if (requireNamespace("z3r", quietly = TRUE)) {
#'   p <- rbind(c(a = 0.9, b = 0.1), c(a = 0.6, b = 0.4), c(a = 0.2, b = 0.8))
#'   ei <- rbind(c(1, 2, 2, 3), c(2, 1, 3, 2))
#'   consistent_labels(p, ei, forbidden = cbind("a", "a"))
#' }
#' @export
consistent_labels <- function(probs, edge_index, forbidden, fixed = NULL, precision = 1000,
                              timeout = 10000) {
  if (!requireNamespace("z3r", quietly = TRUE)) {
    stopf("consistent_labels() needs the z3r package: remotes::install_github(\"adamrsophie/z3r\").")
  }
  probs <- as.matrix(probs)
  classes <- colnames(probs)
  if (is.null(classes)) stopf("`probs` needs column names (the classes).")
  n <- nrow(probs)
  K <- length(classes)
  edge_index <- matrix(as.integer(edge_index), nrow = 2L)
  if (any(edge_index < 1L | edge_index > n)) stopf("`edge_index` refers to nodes outside 1:%d.", n)
  forbidden <- as.matrix(forbidden)
  if (ncol(forbidden) != 2L || !all(forbidden %in% classes)) {
    stopf("`forbidden` must be a two column table of class names.")
  }
  w <- round(log(pmax(probs, 1e-12)) * precision)
  ctx <- z3r::z3_context(timeout = as.integer(timeout))
  opt <- z3r::z3_optimize(ctx)
  lab <- lapply(seq_len(n), function(i) z3r::z3_int(ctx, paste0("c", i)))
  for (i in seq_len(n)) z3r::z3_add(opt, lab[[i]] >= 1L, lab[[i]] <= K)
  fi <- match(forbidden[, 1L], classes)
  fj <- match(forbidden[, 2L], classes)
  for (e in seq_len(ncol(edge_index))) {
    s <- edge_index[1L, e]
    d <- edge_index[2L, e]
    for (r in seq_along(fi)) z3r::z3_add(opt, !((lab[[s]] == fi[r]) & (lab[[d]] == fj[r])))
  }
  if (!is.null(fixed)) {
    idx <- as.integer(names(fixed))
    for (k in seq_along(idx)) z3r::z3_add(opt, lab[[idx[k]]] == match(as.character(fixed[k]), classes))
  }
  terms <- unlist(lapply(seq_len(n), function(i) lapply(seq_len(K), function(k) {
    z3r::z3_ite(lab[[i]] == k, as.integer(w[i, k]), 0L)
  })), recursive = FALSE)
  z3r::z3_maximize(opt, z3r::z3_sum(terms))
  status <- as.character(z3r::z3_check(opt))
  best <- max.col(probs, ties.method = "first")
  if (status != "sat") {
    return(list(labels = NULL, status = status, changed = integer(0), log_prob = NA_real_))
  }
  m <- z3r::z3_model(opt)
  chosen <- vapply(lab, function(v) as.integer(z3r::z3_value(z3r::z3_eval(m, v))), 1L)
  list(labels = factor(classes[chosen], levels = classes), status = status,
       changed = which(chosen != best),
       log_prob = sum(log(pmax(probs[cbind(seq_len(n), chosen)], 1e-12))))
}

#' Budgeted link selection with optcompR
#'
#' Chooses which candidate links to add to a graph: at most `budget` pairs,
#' each node gaining at most `max_new_degree` new links, maximizing the sum
#' of predicted scores (for example link probabilities from a trained link
#' prediction model). The selection is a 0-1 integer program
#' \deqn{\max_{z \in \{0,1\}^P} \sum_p s_p z_p \quad\text{s.t.}\quad
#'   \sum_p z_p \le B,\ \sum_{p \ni v} z_p \le d_v,}
#' built and solved with
#' [optcompR](https://github.com/adamrsophie/optcompR) (HiGHS by default).
#' Pairs with a score at or below `min_score` are never chosen.
#'
#' This is an optional bridge used after training; integer programs are not
#' differentiable and are not part of any model.
#'
#' @param pairs Integer matrix, 2 by P, of candidate node pairs (1-based).
#' @param scores Numeric score per pair.
#' @param budget Maximum number of pairs.
#' @param max_new_degree Maximum new links per node (a scalar or one value
#'   per node, indexed by node number).
#' @param min_score Pairs scoring at or below this are excluded.
#' @return A list with `selected` (column indices into `pairs`), `pairs`
#'   (the selected pairs), `objective` and `status`.
#' @examples
#' if (requireNamespace("optcompR", quietly = TRUE)) {
#'   pairs <- rbind(c(1, 1, 2, 3), c(2, 3, 3, 4))
#'   select_links(pairs, c(0.9, 0.8, 0.7, 0.6), budget = 3, max_new_degree = 1)
#' }
#' @export
select_links <- function(pairs, scores, budget, max_new_degree = Inf, min_score = -Inf) {
  if (!requireNamespace("optcompR", quietly = TRUE)) {
    stopf("select_links() needs the optcompR package: remotes::install_github(\"adamrsophie/optcompR\").")
  }
  pairs <- matrix(as.integer(pairs), nrow = 2L)
  scores <- as.numeric(scores)
  if (length(scores) != ncol(pairs)) stopf("`scores` must have one value per pair.")
  budget <- check_count(budget, "budget", 0L)
  cand <- which(scores > min_score)
  empty <- list(selected = integer(0), pairs = pairs[, 0L, drop = FALSE], objective = 0,
                status = "optimal")
  if (!length(cand) || budget == 0L) return(empty)
  P <- length(cand)
  nodes <- sort(unique(as.vector(pairs[, cand])))
  inc <- Matrix::sparseMatrix(i = c(match(pairs[1L, cand], nodes), match(pairs[2L, cand], nodes)),
                              j = c(seq_len(P), seq_len(P)), x = 1, dims = c(length(nodes), P))
  cap <- if (length(max_new_degree) == 1L) rep(max_new_degree, length(nodes)) else max_new_degree[nodes]
  m <- optcompR::opt_model("max")
  z <- optcompR::opt_variable(m, "z", dim = P, type = "binary")
  optcompR::opt_objective(m, sum(scores[cand] * z))
  optcompR::opt_constraint(m, sum(z) <= budget)
  lim <- is.finite(cap)
  if (any(lim)) optcompR::opt_constraint(m, inc[lim, , drop = FALSE] %*% z <= cap[lim])
  r <- optcompR::opt_solve(m)
  status <- optcompR::opt_status(r)
  if (status != "optimal") return(list(selected = integer(0), pairs = NULL, objective = NA_real_,
                                       status = status))
  pick <- cand[round(optcompR::opt_value(r, z)) == 1]
  list(selected = pick, pairs = pairs[, pick, drop = FALSE], objective = sum(scores[pick]),
       status = status)
}

#' Neighborhood aggregation with jaxR
#'
#' Computes the message passing aggregation
#' \eqn{h_i = \mathrm{aggr}_{j \to i} w_{ji} x_j} with
#' [jaxR](https://github.com/adamrsophie/jaxR) array operations
#' (`jnp_take` and `ops_segment_sum`), the same operation that
#' [message_passing()] performs with torch. Because it is written in jaxR,
#' the result can be differentiated with `jaxR::grad()`, vectorized with
#' `jaxR::vmap()` and compiled with `jaxR::jit()` (pass `num_nodes` under
#' `jit()`).
#'
#' This is an optional bridge for experiments with a second array backend;
#' graphlearnR's layers and training remain on torch, and models are not
#' converted between backends. It needs jaxR and its XLA runtime
#' (`jaxR::install_pjrt()`).
#'
#' @param x Node features: an R matrix or a jaxR array, N by F.
#' @param edge_index Integer matrix, 2 by E, 1-based (source, target).
#' @param num_nodes Number of target nodes; default `nrow(x)`.
#' @param aggr `"sum"` or `"mean"`.
#' @param edge_weight Optional weight per edge.
#' @return A jaxR array, `num_nodes` by F; `as.matrix()` converts it.
#' @examples
#' \dontrun{
#' x <- matrix(1:6, 3, 2)
#' ei <- rbind(c(1, 2, 3), c(2, 3, 2))
#' as.matrix(aggregate_jax(x, ei, aggr = "mean"))
#' }
#' @export
aggregate_jax <- function(x, edge_index, num_nodes = NULL, aggr = c("sum", "mean"),
                          edge_weight = NULL) {
  if (!requireNamespace("jaxR", quietly = TRUE)) {
    stopf("aggregate_jax() needs the jaxR package: remotes::install_github(\"adamrsophie/jaxR\").")
  }
  aggr <- match.arg(aggr)
  edge_index <- matrix(as.integer(edge_index), nrow = 2L)
  n <- as.integer(num_nodes %||% nrow(x))
  xj <- jaxR::as_jax(x)
  src0 <- jaxR::as_jax(edge_index[1L, ] - 1L)
  dst0 <- jaxR::as_jax(edge_index[2L, ] - 1L)
  msg <- jaxR::jnp_take(xj, src0, axis = 0L)
  if (!is.null(edge_weight)) {
    msg <- msg * jaxR::jnp_reshape(jaxR::as_jax(as.numeric(edge_weight)), c(-1L, 1L))
  }
  out <- jaxR::ops_segment_sum(msg, dst0, num_segments = n)
  if (aggr == "mean") {
    cnt <- jaxR::ops_segment_sum(jaxR::jnp_ones(ncol(edge_index)), dst0, num_segments = n)
    out <- out / jaxR::jnp_reshape(jaxR::jnp_maximum(cnt, 1), c(-1L, 1L))
  }
  out
}

#' Training node models with luz
#'
#' Helpers to train a node level graph model with
#' [luz](https://mlverse.github.io/luz/) instead of [train_graph_model()].
#' `luz_node_dataloader()` returns a dataloader with a single full graph
#' batch whose input is `list(x, edge_index, idx)` (plus `edge_weight` when
#' the graph has one) and whose
#' target is the labels of the nodes in `mask`; `luz_node_module` wraps any
#' model whose forward takes `(x, edge_index, edge_weight = )` so that it
#' returns predictions for those nodes only. Losses are therefore computed
#' on the masked nodes, and message passing still uses the whole graph.
#'
#' Use it with `luz::setup()`, `luz::set_hparams(model = ...)` and
#' `luz::fit()`; see the example. luz handles devices, callbacks and metrics;
#' graphlearnR's own trainer adds neighbor and subgraph sampling, link and
#' graph tasks, and a reproducibility record, which this path does not.
#'
#' @param g A `graph_data` object with node labels and masks.
#' @param mask `"train"`, `"val"` or `"test"`.
#' @param model For `luz_node_module`: a model whose forward takes
#'   `(x, edge_index, edge_weight = )`, for example from [gnn_model()].
#' @return `luz_node_dataloader()`: a torch dataloader. `luz_node_module`: an
#'   `nn_module` generator taking the wrapped `model`.
#' @examples
#' if (torch::torch_is_installed() && requireNamespace("luz", quietly = TRUE)) {
#'   library(luz)
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   fitted <- luz_node_module |>
#'     setup(loss = torch::nn_cross_entropy_loss(), optimizer = torch::optim_adam) |>
#'     set_hparams(model = gnn_model("gcn", 34, 16, 2)) |>
#'     fit(luz_node_dataloader(g, "train"), epochs = 20, verbose = FALSE,
#'         valid_data = luz_node_dataloader(g, "val"),
#'         accelerator = accelerator(cpu = TRUE))
#' }
#' @export
luz_node_dataloader <- function(g, mask = c("train", "val", "test")) {
  mask <- match.arg(mask)
  m <- g[[paste0(mask, "_mask")]]
  if (is.null(m)) stopf("The graph has no `%s_mask`; use split_nodes() first.", mask)
  if (is.null(g$y)) stopf("The graph has no node labels `y`.")
  tt <- graph_tensors(g)
  idx <- tensor_which(tt[[paste0(mask, "_mask")]])
  input <- list(x = tt$x, edge_index = tt$edge_index, idx = idx)
  # NULL entries do not survive the dataloader, so the weight is added only when present.
  if (!is.null(tt$edge_weight)) input$edge_weight <- tt$edge_weight
  item <- list(input, tt$y$index_select(1L, idx))
  ds <- torch::dataset(
    name = "graph_node_batch",
    initialize = function() NULL,
    .getitem = function(i) item,
    .length = function() 1L
  )
  torch::dataloader(ds(), batch_size = NULL)
}

#' @rdname luz_node_dataloader
#' @export
luz_node_module <- nn_module(
  "luz_node_module",
  initialize = function(model) {
    self$model <- model
  },
  forward = function(input) {
    out <- self$model(input$x, input$edge_index, edge_weight = input$edge_weight)
    out$index_select(1L, input$idx)
  }
)
