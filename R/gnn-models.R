#' Build a graph neural network
#'
#' Stacks `num_layers` message passing layers of one architecture, each
#' followed by an activation, optional normalization and dropout, then adds a
#' prediction head:
#'
#' * node level (`pool = NULL`): a linear layer to `output_dim` per node;
#' * graph level (`pool` set): global pooling, then an MLP to `output_dim`
#'   per graph;
#' * encoder only (`output_dim = NULL`): returns node embeddings of size
#'   `hidden_dim`, for link prediction or representation learning.
#'
#' The model's `forward(x, edge_index, edge_weight, edge_attr, batch,
#' num_graphs, edge_type)` takes tensors from [graph_tensors()]. Setting
#' `return_embedding = TRUE` returns the last hidden node representation
#' instead of the head output. The construction arguments are kept in
#' `model$config` so that [save_graph_fit()] can rebuild the model.
#'
#' @param architecture One of `"sage"`, `"gcn"`, `"gat"`, `"gatv2"`,
#'   `"gin"`, `"graph_conv"`, `"transformer"`, `"mpnn"` (edge conditioned
#'   [nn_conv()]), `"rgcn"`.
#' @param input_dim Node feature size.
#' @param hidden_dim Hidden size. For `"gat"`, `"gatv2"` and
#'   `"transformer"` it must be divisible by `heads`.
#' @param output_dim Output size (classes or targets), or `NULL`.
#' @param num_layers Number of message passing layers.
#' @param dropout Dropout probability after each layer.
#' @param activation Activation module constructor.
#' @param heads Attention heads.
#' @param edge_dim Edge feature size (`"mpnn"`, GINE, attention with edge
#'   features).
#' @param aggr Neighbor aggregation for `"sage"`, `"graph_conv"`, `"mpnn"`,
#'   `"rgcn"`.
#' @param pool `NULL`, `"mean"`, `"sum"`, `"max"` or `"attention"`.
#' @param norm `"none"`, `"batch"` or `"layer"` normalization after each
#'   layer.
#' @param residual Add a residual connection around each layer after the
#'   first.
#' @param num_relations,num_bases Relation count and bases for `"rgcn"`.
#' @return An `nn_module` of class `graph_model`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   m <- gnn_model("sage", input_dim = 4, hidden_dim = 8, output_dim = 3)
#'   x <- torch::torch_randn(5, 4)
#'   ei <- torch::torch_tensor(rbind(1:4, 2:5), dtype = torch::torch_long())
#'   m(x, ei)$shape
#' }
#' @export
gnn_model <- function(architecture = c("sage", "gcn", "gat", "gatv2", "gin",
                                       "graph_conv", "transformer", "mpnn",
                                       "rgcn"),
                      input_dim, hidden_dim = 64L, output_dim = NULL,
                      num_layers = 2L, dropout = 0.5, activation = "relu",
                      heads = 4L, edge_dim = NULL, aggr = NULL, pool = NULL,
                      norm = c("none", "batch", "layer"), residual = FALSE,
                      num_relations = NULL, num_bases = NULL) {
  architecture <- match.arg(architecture)
  norm <- match.arg(norm)
  check_count(input_dim, "input_dim", 1L)
  check_count(hidden_dim, "hidden_dim", 1L)
  check_count(num_layers, "num_layers", 1L)
  if (!is.null(output_dim)) check_count(output_dim, "output_dim", 1L)
  if (!is.null(pool)) pool <- match.arg(pool, c("mean", "sum", "max", "attention"))
  if (architecture %in% c("gat", "gatv2", "transformer") && hidden_dim %% heads != 0L) {
    stopf("`hidden_dim` (%d) must be divisible by `heads` (%d).", hidden_dim, heads)
  }
  if (architecture == "mpnn" && is.null(edge_dim)) {
    stopf("The \"mpnn\" architecture needs `edge_dim`.")
  }
  if (architecture == "rgcn" && is.null(num_relations)) {
    stopf("The \"rgcn\" architecture needs `num_relations`.")
  }
  if (!is.character(activation)) stopf("`activation` must be a name such as \"relu\".")
  config <- list(architecture = architecture, input_dim = as.integer(input_dim),
                 hidden_dim = as.integer(hidden_dim),
                 output_dim = if (!is.null(output_dim)) as.integer(output_dim),
                 num_layers = as.integer(num_layers), dropout = dropout,
                 activation = activation, heads = as.integer(heads),
                 edge_dim = edge_dim, aggr = aggr, pool = pool, norm = norm,
                 residual = residual, num_relations = num_relations,
                 num_bases = num_bases)
  graph_model_module(config)
}

make_conv <- function(arch, d_in, d_out, cfg) {
  aggr <- cfg$aggr
  switch(arch,
    sage = sage_conv(d_in, d_out, aggr = aggr %||% "mean"),
    gcn = gcn_conv(d_in, d_out),
    graph_conv = graph_conv(d_in, d_out, aggr = aggr %||% "sum"),
    gin = gin_conv(graph_mlp(c(d_in, d_out, d_out)), train_eps = TRUE,
                   edge_dim = cfg$edge_dim, in_features = d_in),
    gat = gat_conv(d_in, d_out %/% cfg$heads, heads = cfg$heads,
                   edge_dim = cfg$edge_dim),
    gatv2 = gatv2_conv(d_in, d_out %/% cfg$heads, heads = cfg$heads,
                       edge_dim = cfg$edge_dim),
    transformer = transformer_conv(d_in, d_out %/% cfg$heads, heads = cfg$heads,
                                   edge_dim = cfg$edge_dim),
    mpnn = nn_conv(d_in, d_out, edge_dim = cfg$edge_dim, aggr = aggr %||% "mean"),
    rgcn = rgcn_conv(d_in, d_out, num_relations = cfg$num_relations,
                     num_bases = cfg$num_bases, aggr = aggr %||% "mean")
  )
}

activation_fn <- function(name) {
  switch(name,
    relu = nnf_relu,
    elu = nnf_elu,
    gelu = nnf_gelu,
    tanh = torch_tanh,
    leaky_relu = function(x) nnf_leaky_relu(x, 0.2),
    stopf("Unknown activation `%s`.", name)
  )
}

graph_model_module <- nn_module(
  "graph_model",
  initialize = function(config) {
    self$config <- config
    h <- config$hidden_dim
    self$convs <- nn_module_list(lapply(seq_len(config$num_layers), function(i) {
      make_conv(config$architecture, if (i == 1L) config$input_dim else h, h, config)
    }))
    if (config$norm != "none") {
      self$norms <- nn_module_list(lapply(seq_len(config$num_layers), function(i) {
        if (config$norm == "batch") nn_batch_norm1d(h) else nn_layer_norm(h)
      }))
    }
    if (!is.null(config$pool)) {
      if (config$pool == "attention") self$att_pool <- global_attention_pool(h)
      if (!is.null(config$output_dim)) {
        self$head <- graph_mlp(c(h, h, config$output_dim), dropout = config$dropout)
      }
    } else if (!is.null(config$output_dim)) {
      self$head <- nn_linear(h, config$output_dim)
    }
  },
  forward = function(x, edge_index, edge_weight = NULL, edge_attr = NULL,
                     batch = NULL, num_graphs = NULL, edge_type = NULL,
                     return_embedding = FALSE, ...) {
    act <- activation_fn(self$config$activation)
    h <- x
    for (i in seq_along(self$convs)) {
      conv <- self$convs[[i]]
      out <- if (self$config$architecture == "rgcn") {
        conv(h, edge_index, edge_type = edge_type)
      } else {
        conv(h, edge_index, edge_weight = edge_weight, edge_attr = edge_attr)
      }
      if (!is.null(self$norms)) out <- self$norms[[i]](out)
      out <- act(out)
      if (self$config$residual && i > 1L) out <- out + h
      h <- nnf_dropout(out, p = self$config$dropout, training = self$training)
    }
    if (return_embedding || is.null(self$head) && is.null(self$config$pool)) {
      return(h)
    }
    if (!is.null(self$config$pool)) {
      h <- switch(self$config$pool,
        mean = global_mean_pool(h, batch, num_graphs),
        sum = global_add_pool(h, batch, num_graphs),
        max = global_max_pool(h, batch, num_graphs),
        attention = self$att_pool(h, batch, num_graphs)
      )
      if (is.null(self$head)) return(h)
    }
    self$head(h)
  }
)

# Call a model on graph tensors, passing the standard arguments by name.
model_forward <- function(model, tt, ...) {
  model(tt$x, tt$edge_index, edge_weight = tt$edge_weight,
        edge_attr = tt$edge_attr, batch = tt$batch,
        num_graphs = tt$num_graphs, edge_type = tt$edge_type, ...)
}

#' Link and edge decoders
#'
#' `link_decoder()` scores node pairs from node embeddings `z`:
#'
#' * `"dot"`: \eqn{z_u^\top z_v};
#' * `"distmult"`: \eqn{z_u^\top \mathrm{diag}(r) z_v} with one learned
#'   vector per relation (DistMult, Yang et al. 2015);
#' * `"mlp"`: an MLP on \eqn{[z_u \Vert z_v]} (order sensitive, so suited to
#'   directed links);
#' * `"hadamard"`: an MLP on \eqn{z_u \odot z_v}.
#'
#' Scores are logits; apply a sigmoid for probabilities. `edge_decoder()` is
#' an MLP on \eqn{[z_u \Vert z_v \Vert e_{uv}]} for edge classification or
#' regression.
#'
#' @param type Decoder type.
#' @param dim Embedding size (not needed for `"dot"`).
#' @param num_relations Relations for `"distmult"`.
#' @param hidden Hidden size of MLP decoders.
#' @return An `nn_module` called as `decoder(z, pairs, relation = NULL)`,
#'   where `pairs` is a 2 by K long tensor of one based node indices. For
#'   typed links pass `z = list(z_src, z_dst)`.
#' @export
link_decoder <- nn_module(
  "link_decoder",
  initialize = function(type = c("dot", "distmult", "mlp", "hadamard"),
                        dim = NULL, num_relations = 1L, hidden = 64L) {
    self$type <- match.arg(type)
    if (self$type != "dot" && is.null(dim)) stopf("This decoder needs `dim`.")
    if (self$type == "distmult") {
      self$rel <- nn_parameter(torch_ones(num_relations, dim) + 0.1 * torch_randn(num_relations, dim))
    }
    if (self$type == "mlp") self$mlp <- graph_mlp(c(2L * dim, hidden, 1L))
    if (self$type == "hadamard") self$mlp <- graph_mlp(c(dim, hidden, 1L))
  },
  forward = function(z, pairs, relation = NULL) {
    zu <- src_of(z)$index_select(1L, pairs[1L, ])
    zv <- dst_of(z)$index_select(1L, pairs[2L, ])
    switch(self$type,
      dot = (zu * zv)$sum(dim = -1L),
      distmult = {
        r <- if (is.null(relation)) self$rel[1L, ]$unsqueeze(1L)
             else self$rel$index_select(1L, as_index_tensor(relation, zu$device))
        (zu * r * zv)$sum(dim = -1L)
      },
      mlp = self$mlp(torch_cat(list(zu, zv), dim = -1L))$squeeze(-1L),
      hadamard = self$mlp(zu * zv)$squeeze(-1L)
    )
  }
)

#' @rdname link_decoder
#' @param out_dim Output size of `edge_decoder()`.
#' @param edge_dim Edge feature size appended to the pair (0 for none).
#' @export
edge_decoder <- nn_module(
  "edge_decoder",
  initialize = function(dim, out_dim, edge_dim = 0L, hidden = 64L) {
    self$edge_dim <- edge_dim
    self$mlp <- graph_mlp(c(2L * dim + edge_dim, hidden, out_dim))
  },
  forward = function(z, pairs, edge_attr = NULL) {
    parts <- list(src_of(z)$index_select(1L, pairs[1L, ]), dst_of(z)$index_select(1L, pairs[2L, ]))
    if (self$edge_dim > 0L) parts[[3L]] <- edge_attr
    self$mlp(torch_cat(parts, dim = -1L))
  }
)
