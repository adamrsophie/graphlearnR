#' Heterogeneous graph transformer convolution
#'
#' One layer of the heterogeneous graph transformer (HGT, Hu et al. 2020).
#' For a target node \eqn{t} of type \eqn{\tau(t)}, an incoming edge from
#' \eqn{s} through relation \eqn{r}, and head \eqn{h} with \eqn{D} channels,
#' \deqn{k_s = K_{\tau(s)} x_s,\ q_t = Q_{\tau(t)} x_t,\ v_s = V_{\tau(s)} x_s,}
#' \deqn{e_{st}^h = (k_s^h W^{att}_{r,h}) \cdot q_t^h \ \mu_{r,h} / \sqrt{D},}
#' \deqn{\alpha_{st}^h = \mathrm{softmax}_{s \in N(t)}(e_{st}^h),}
#' where the softmax runs over **all** incoming edges of \eqn{t}, across
#' relations. The update is
#' \deqn{\tilde h_t = \sum_s \alpha_{st} (v_s W^{msg}_{r}),\quad
#'   h_t' = \beta_{\tau(t)} A_{\tau(t)}(\mathrm{GELU}(\tilde h_t)) +
#'   (1 - \beta_{\tau(t)}) x_t,}
#' with \eqn{\beta = \mathrm{sigmoid}(\text{skip})} a learned gate per type
#' (the skip term is used only when input and output sizes match). Node types
#' without incoming edges get \eqn{A_{\tau}(0)} plus the skip term.
#' Relative temporal encoding from the paper is not implemented.
#'
#' Cost is \eqn{O(E H D^2)} for the relation transforms plus
#' \eqn{O(N F F')} for the type projections; the layer keeps an E by heads
#' attention tensor per target type in `last_attention`.
#'
#' @param node_types Character vector of node types.
#' @param relations Character vector of relation keys `"src__rel__dst"`.
#' @param in_features Input size, the same for every node type.
#' @param out_features Output size; must be divisible by `heads`.
#' @param heads Number of attention heads.
#' @return An `nn_module` called as `layer(x_dict, edge_index_dict)` that
#'   returns a named list of node feature tensors.
#' @references Hu, Dong, Wang and Sun (2020) Heterogeneous graph transformer.
#'   Proceedings of The Web Conference.
#' @examples
#' if (torch::torch_is_installed()) {
#'   hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
#'   ht <- hetero_tensors(hg)
#'   layer <- hgt_conv(hg$node_types, names(hg$edges), 8, 8, heads = 2)
#'   x <- list(drug = torch::torch_randn(60, 8), protein = torch::torch_randn(80, 8))
#'   lapply(layer(x, ht$edge_index), dim)
#' }
#' @export
hgt_conv <- nn_module(
  "hgt_conv",
  initialize = function(node_types, relations, in_features, out_features, heads = 2L) {
    heads <- check_count(heads, "heads", 1L)
    if (out_features %% heads != 0L) stopf("`out_features` must be divisible by `heads`.")
    self$node_types <- node_types
    self$relations <- relations
    self$heads <- heads
    self$dim <- as.integer(out_features %/% heads)
    self$in_features <- in_features
    self$out_features <- out_features
    per_type <- function(f) nn_module_dict(stats::setNames(lapply(node_types, function(t) f()), node_types))
    self$k_lin <- per_type(function() nn_linear(in_features, out_features))
    self$q_lin <- per_type(function() nn_linear(in_features, out_features))
    self$v_lin <- per_type(function() nn_linear(in_features, out_features))
    self$a_lin <- per_type(function() nn_linear(out_features, out_features))
    self$skip <- nn_parameter(torch_ones(length(node_types)))
    R <- length(relations)
    D <- self$dim
    eye <- torch_eye(D)$unsqueeze(1L)$unsqueeze(1L)$expand(c(R, heads, D, D))
    self$w_att <- nn_parameter(eye + 0.1 * torch_randn(R, heads, D, D))
    self$w_msg <- nn_parameter(eye + 0.1 * torch_randn(R, heads, D, D))
    self$mu <- nn_parameter(torch_ones(R, heads))
    self$last_attention <- NULL
  },
  forward = function(x_dict, edge_index_dict) {
    H <- self$heads
    D <- self$dim
    proj <- function(lin) {
      lapply(stats::setNames(self$node_types, self$node_types), function(t) {
        if (is.null(x_dict[[t]])) NULL else lin[[t]](x_dict[[t]])$view(c(-1L, H, D))
      })
    }
    k <- proj(self$k_lin)
    q <- proj(self$q_lin)
    v <- proj(self$v_lin)
    out <- list()
    attn <- list()
    for (ti in seq_along(self$node_types)) {
      d <- self$node_types[ti]
      if (is.null(x_dict[[d]])) next
      n_d <- x_dict[[d]]$size(1L)
      scores <- list()
      msgs <- list()
      dsts <- list()
      for (r in seq_along(self$relations)) {
        key <- self$relations[r]
        parts <- strsplit(key, "__", fixed = TRUE)[[1L]]
        if (parts[3L] != d) next
        ei <- edge_index_dict[[key]]
        s <- parts[1L]
        if (is.null(ei) || ei$size(2L) == 0L || is.null(k[[s]])) next
        src <- ei[1L, ]
        dst <- ei[2L, ]
        kj <- torch_einsum("ehd,hdf->ehf", list(k[[s]]$index_select(1L, src), self$w_att[r, , , ]))
        vj <- torch_einsum("ehd,hdf->ehf", list(v[[s]]$index_select(1L, src), self$w_msg[r, , , ]))
        qi <- q[[d]]$index_select(1L, dst)
        scores[[length(scores) + 1L]] <- (kj * qi)$sum(dim = -1L) * self$mu[r, ] / sqrt(D)
        msgs[[length(msgs) + 1L]] <- vj
        dsts[[length(dsts) + 1L]] <- dst
      }
      xd <- x_dict[[d]]
      if (length(scores)) {
        sc <- torch_cat(scores, dim = 1L)
        dst <- torch_cat(dsts)
        alpha <- scatter_softmax(sc, dst, n_d)
        msg <- (torch_cat(msgs, dim = 1L) * alpha$unsqueeze(-1L))$reshape(c(-1L, H * D))
        agg <- scatter_sum(msg, dst, n_d)
        attn[[d]] <- list(alpha = alpha$detach(), index = dst)
      } else {
        agg <- torch_zeros(n_d, H * D, dtype = xd$dtype, device = xd$device)
      }
      a <- self$a_lin[[d]](nnf_gelu(agg))
      out[[d]] <- if (self$in_features == self$out_features) {
        beta <- torch_sigmoid(self$skip[ti])
        beta * a + (1 - beta) * xd
      } else {
        a
      }
    }
    self$last_attention <- attn
    out
  }
)
