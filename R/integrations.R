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
