#' Continuous depth graph diffusion (graph neural ODE)
#'
#' `graph_ode_block()` evolves node features by the graph diffusion equation
#' of GRAND (Chamberlain et al. 2021),
#' \deqn{\frac{dX}{dt} = (A - I) X,\quad X(0) = X_0,}
#' from \eqn{t = 0} to `t_end`, and returns \eqn{X(t_{end})}.
#'
#' * `diffusion = "gcn"`: \eqn{A = \hat D^{-1/2}(A + I)\hat D^{-1/2}}, the GCN
#'   propagation matrix (linear, no parameters in the dynamics).
#' * `diffusion = "attention"`: \eqn{A_{ij} = \mathrm{softmax}_j((W_K x_j)^\top
#'   W_Q x_i / \sqrt{d})} over the incoming edges of \eqn{i} and a self loop,
#'   computed once from \eqn{X_0} (GRAND-l).
#'
#' The equation is integrated with `steps` fixed steps of explicit Euler or
#' classical fourth order Runge-Kutta inside the torch graph, so gradients
#' flow by ordinary backpropagation through the steps. This is a fixed
#' discretization that defines the layer, not a general ODE solver: there is
#' no step size control, adjoint method or error estimate. For adaptive
#' simulation of graph dynamics outside training, use [graph_diffusion()],
#' which calls modelcompR.
#'
#' `graph_ode_model()` wraps the block between a linear encoder and decoder
#' for node level tasks and can be passed to [train_graph_model()].
#'
#' @param dim Feature size (unchanged by the block).
#' @param diffusion `"gcn"` or `"attention"`.
#' @param t_end Integration time.
#' @param steps Number of integration steps.
#' @param method `"rk4"` or `"euler"`.
#' @return An `nn_module`.
#' @references Chamberlain, Rowbottom, Gorinova, Webb, Rossi and Bronstein
#'   (2021) GRAND: graph neural diffusion. Chen, Rubanova, Bettencourt and
#'   Duvenaud (2018) Neural ordinary differential equations.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   m <- graph_ode_model(34, 16, 2, t_end = 3, steps = 6)
#'   fit <- train_graph_model(m, g, "node_classification", epochs = 50, seed = 1)
#'   evaluate_graph_model(fit, g)
#' }
#' @export
graph_ode_block <- nn_module(
  "graph_ode_block",
  initialize = function(dim, diffusion = c("gcn", "attention"), t_end = 1, steps = 4L,
                        method = c("rk4", "euler")) {
    self$diffusion <- match.arg(diffusion)
    self$method <- match.arg(method)
    self$t_end <- t_end
    self$steps <- check_count(steps, "steps", 1L)
    if (self$diffusion == "attention") {
      self$key <- nn_linear(dim, dim, bias = FALSE)
      self$query <- nn_linear(dim, dim, bias = FALSE)
    }
  },
  operator = function(x, edge_index, edge_weight = NULL) {
    n <- x$size(1L)
    if (self$diffusion == "gcn") return(gcn_norm(edge_index, n, edge_weight))
    sl <- add_self_loops_tensor(remove_self_loops_tensor(edge_index)$edge_index, n)
    ei <- sl$edge_index
    k <- self$key(x)$index_select(1L, ei[1L, ])
    q <- self$query(x)$index_select(1L, ei[2L, ])
    w <- scatter_softmax((k * q)$sum(dim = 2L) / sqrt(x$size(2L)), ei[2L, ], n)
    list(edge_index = ei, edge_weight = w)
  },
  forward = function(x, edge_index, edge_weight = NULL, ...) {
    op <- self$operator(x, edge_index, edge_weight)
    src <- op$edge_index[1L, ]
    dst <- op$edge_index[2L, ]
    n <- x$size(1L)
    f <- function(h) {
      scatter_sum(h$index_select(1L, src) * op$edge_weight$unsqueeze(2L), dst, n) - h
    }
    dt <- self$t_end / self$steps
    h <- x
    for (s in seq_len(self$steps)) {
      h <- if (self$method == "euler") {
        h + dt * f(h)
      } else {
        k1 <- f(h)
        k2 <- f(h + dt / 2 * k1)
        k3 <- f(h + dt / 2 * k2)
        k4 <- f(h + dt * k3)
        h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
      }
    }
    h
  }
)

#' @rdname graph_ode_block
#' @param input_dim,hidden_dim,output_dim Feature sizes.
#' @param dropout Dropout on the encoder output.
#' @export
graph_ode_model <- nn_module(
  "graph_ode_model",
  initialize = function(input_dim, hidden_dim = 32L, output_dim = 2L,
                        diffusion = c("gcn", "attention"), t_end = 1, steps = 4L,
                        method = c("rk4", "euler"), dropout = 0) {
    self$encoder <- nn_linear(input_dim, hidden_dim)
    self$ode <- graph_ode_block(hidden_dim, diffusion, t_end, steps, method)
    self$decoder <- nn_linear(hidden_dim, output_dim)
    self$dropout <- dropout
    self$config <- list(architecture = "graph_ode", input_dim = input_dim,
                        hidden_dim = hidden_dim, output_dim = output_dim)
  },
  forward = function(x, edge_index, edge_weight = NULL, return_embedding = FALSE, ...) {
    h <- nnf_dropout(self$encoder(x), p = self$dropout, training = self$training)
    h <- self$ode(h, edge_index, edge_weight)
    if (return_embedding) h else self$decoder(h)
  }
)
