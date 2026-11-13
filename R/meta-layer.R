#' Edge, node and graph update block
#'
#' One block of the graph network of Battaglia et al. (2018), in the form of
#' PyTorch Geometric's `MetaLayer`: an edge update, then a node update, then
#' a graph level update, each optional. With edges \eqn{(j \to i)}, node
#' features \eqn{x}, edge features \eqn{e} and graph features \eqn{u}:
#'
#' 1. edge update \eqn{e'_{ji} = \phi^e(x_j, x_i, e_{ji}, u_{g(i)})};
#' 2. node update \eqn{x'_i = \phi^v(x, \mathrm{edge\_index}, e', u, \mathrm{batch})}
#'    (typically an aggregation of the updated incoming edges, see
#'    [message_passing()] and [scatter_aggregate()]);
#' 3. graph update \eqn{u'_g = \phi^u(x', \mathrm{edge\_index}, e', u, \mathrm{batch})}
#'    (typically a pooled summary, see [global-pooling]).
#'
#' Each update is a function or `nn_module` called with named arguments:
#'
#' * `edge_model(x_src, x_dst, edge_attr, u, batch)`: `x_src` and `x_dst`
#'   are the source and target node features gathered per edge, `u` the
#'   graph features gathered per edge (`NULL` without `u`), `batch` the
#'   graph of each edge. Returns E rows.
#' * `node_model(x, edge_index, edge_attr, u, batch)`: returns N rows.
#' * `global_model(x, edge_index, edge_attr, u, batch)`: returns one row per
#'   graph.
#'
#' The block is permutation equivariant in nodes and edges (and invariant
#' for the graph output) whenever the update functions work row wise and
#' aggregate with permutation invariant operators.
#'
#' @param edge_model,node_model,global_model Update functions or modules, or
#'   `NULL` to keep the corresponding features unchanged.
#' @return An `nn_module` called as
#'   `block(x, edge_index, edge_attr = NULL, u = NULL, batch = NULL)`,
#'   returning a list with `x`, `edge_attr` and `u`.
#' @references Battaglia et al. (2018) Relational inductive biases, deep
#'   learning, and graph networks. arXiv:1806.01261.
#' @examples
#' if (torch::torch_is_installed()) {
#'   edge_mlp <- torch::nn_linear(2 * 3 + 1, 4)
#'   node_mlp <- torch::nn_linear(3 + 4, 3)
#'   block <- meta_layer(
#'     edge_model = function(x_src, x_dst, edge_attr, ...) {
#'       edge_mlp(torch::torch_cat(list(x_src, x_dst, edge_attr), dim = 2))
#'     },
#'     node_model = function(x, edge_index, edge_attr, ...) {
#'       agg <- scatter_mean(edge_attr, edge_index[2, ], x$size(1))
#'       node_mlp(torch::torch_cat(list(x, agg), dim = 2))
#'     },
#'     global_model = function(x, batch, ...) global_mean_pool(x, batch)
#'   )
#'   g <- dataset_karate()
#'   tt <- graph_tensors(g)
#'   out <- block(torch::torch_randn(34, 3), tt$edge_index,
#'                edge_attr = torch::torch_randn(num_edges(g), 1))
#'   lapply(out, function(t) t$shape)
#' }
#' @export
meta_layer <- nn_module(
  "meta_layer",
  initialize = function(edge_model = NULL, node_model = NULL, global_model = NULL) {
    # Modules are registered so their parameters are trained; plain
    # functions are kept as they are.
    self$edge_model <- edge_model
    self$node_model <- node_model
    self$global_model <- global_model
  },
  forward = function(x, edge_index, edge_attr = NULL, u = NULL, batch = NULL) {
    if (!is.null(batch)) batch <- as_index_tensor(batch, x$device)
    src <- edge_index[1L, ]
    dst <- edge_index[2L, ]
    if (!is.null(self$edge_model)) {
      ebatch <- if (is.null(batch)) NULL else batch$index_select(1L, dst)
      u_e <- if (is.null(u)) NULL
             else if (is.null(ebatch)) u$expand(c(src$size(1L), -1L))
             else u$index_select(1L, ebatch)
      edge_attr <- self$edge_model(x_src = x$index_select(1L, src), x_dst = x$index_select(1L, dst),
                                   edge_attr = edge_attr, u = u_e, batch = ebatch)
    }
    if (!is.null(self$node_model)) {
      x <- self$node_model(x = x, edge_index = edge_index, edge_attr = edge_attr, u = u, batch = batch)
    }
    if (!is.null(self$global_model)) {
      u <- self$global_model(x = x, edge_index = edge_index, edge_attr = edge_attr, u = u, batch = batch)
    }
    list(x = x, edge_attr = edge_attr, u = u)
  }
)
