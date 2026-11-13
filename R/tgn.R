#' Temporal graph network with node memory
#'
#' A memory based model for continuous time dynamic link prediction in the
#' style of TGN (Rossi et al. 2020). Every node keeps a memory vector that a
#' GRU updates whenever the node takes part in an event. Events are processed
#' in chronological batches:
#'
#' 1. The raw messages of the **previous** batch update the memory of the
#'    nodes involved (message \eqn{[s_i, s_j, \phi(t - t_i^{last}), e_{ij}]},
#'    keeping the most recent message per node, then
#'    \eqn{s_i \leftarrow \mathrm{GRU}(m_i, s_i)}). Doing this at the start
#'    of the next batch lets gradients reach the GRU while the current
#'    batch's own events never inform its predictions.
#' 2. Node embeddings are computed from memory and the time since the node's
#'    last update, \eqn{z_i = W [s_i, \phi(t - t_i^{last})]}
#'    (`embedding = "time"`), plus, with `embedding = "graph_sum"`, the mean
#'    of \eqn{W' [s_j, \phi(t - t_{ij})]} over the `num_neighbors` most recent
#'    earlier interaction partners \eqn{j}. With
#'    `embedding = "graph_attention"` (the paper's best variant), multi head
#'    attention runs over those partners with query
#'    \eqn{W_Q [s_i, \phi(0)]}, keys and values
#'    \eqn{W_{K,V} [s_j, \phi(t - t_{ij}), e_{ij}]}, and
#'    \eqn{z_i = \mathrm{MLP}([\tilde h_i, s_i, \phi(t - t_i^{last})])},
#'    \eqn{\tilde h_i} the attention output (zero without partners); the
#'    weights of the last call are kept in `model$last_attention`.
#' 3. An MLP scores \eqn{[z_{src}, z_{dst}]}; the loss is binary cross entropy
#'    against one random destination per event.
#'
#' \eqn{\phi(\Delta t) = \cos(\omega \Delta t + b)} is the learnable time
#' encoding of TGAT (Xu et al. 2020). Memory is detached between batches
#' (truncated backpropagation over one batch). Only one embedding layer
#' (one hop) and the "last" message aggregator are implemented.
#'
#' @param num_nodes Number of nodes.
#' @param memory_dim Memory size.
#' @param time_dim Size of the time encoding.
#' @param embedding_dim Embedding size.
#' @param edge_dim Number of event features (`0` for none).
#' @param embedding `"time"`, `"graph_sum"` or `"graph_attention"`.
#' @param num_neighbors Most recent partners used by the graph embeddings.
#' @param heads Attention heads for `"graph_attention"`; `embedding_dim`
#'   must be divisible by it.
#' @param node_embedding Add a learned static embedding per node to
#'   \eqn{z_i} (useful without node features; makes the model
#'   transductive).
#' @return `tgn_model()`: an `nn_module`.
#' @references Rossi, Chamberlain, Frasca, Eynard, Monti and Bronstein (2020)
#'   Temporal graph networks for deep learning on dynamic graphs. Xu, Ruan,
#'   Korpeoglu, Kumar and Achan (2020) Inductive representation learning on
#'   temporal graphs.
#' @examples
#' if (torch::torch_is_installed()) {
#'   tg <- dataset_synthetic_temporal(seed = 1)
#'   sp <- split_temporal(tg)
#'   m <- tgn_model(tg$num_nodes, memory_dim = 16, embedding_dim = 16)
#'   fit <- train_tgn(m, sp, epochs = 2, batch_size = 100, seed = 1)
#'   fit$test_metrics
#' }
#' @export
tgn_model <- nn_module(
  "tgn_model",
  initialize = function(num_nodes, memory_dim = 32L, time_dim = 16L, embedding_dim = 32L,
                        edge_dim = 0L, embedding = c("time", "graph_sum", "graph_attention"),
                        num_neighbors = 10L, node_embedding = TRUE, heads = 2L) {
    self$num_nodes <- check_count(num_nodes, "num_nodes", 1L)
    self$memory_dim <- memory_dim
    self$edge_dim <- edge_dim
    self$embedding <- match.arg(embedding)
    self$num_neighbors <- check_count(num_neighbors, "num_neighbors", 1L)
    self$time_w <- nn_parameter(torch_tensor(1 / 10^seq(0, 9, length.out = time_dim)))
    self$time_b <- nn_parameter(torch_zeros(time_dim))
    gin <- 2L * memory_dim + time_dim + edge_dim
    self$gru_x <- nn_linear(gin, 3L * memory_dim)
    self$gru_h <- nn_linear(memory_dim, 3L * memory_dim)
    self$emb_self <- nn_linear(memory_dim + time_dim, embedding_dim)
    if (self$embedding == "graph_sum") {
      self$emb_nbr <- nn_linear(memory_dim + time_dim, embedding_dim, bias = FALSE)
    }
    if (self$embedding == "graph_attention") {
      self$heads <- check_count(heads, "heads", 1L)
      if (embedding_dim %% heads != 0L) stopf("`embedding_dim` must be divisible by `heads`.")
      self$att_q <- nn_linear(memory_dim + time_dim, embedding_dim)
      self$att_k <- nn_linear(memory_dim + time_dim + edge_dim, embedding_dim)
      self$att_v <- nn_linear(memory_dim + time_dim + edge_dim, embedding_dim)
      self$att_out <- nn_sequential(nn_linear(embedding_dim + memory_dim + time_dim,
                                              embedding_dim), nn_relu(),
                                    nn_linear(embedding_dim, embedding_dim))
    }
    if (node_embedding) self$static <- nn_embedding(num_nodes, embedding_dim)
    self$decoder <- nn_sequential(nn_linear(2L * embedding_dim, embedding_dim), nn_relu(),
                                  nn_linear(embedding_dim, 1L))
  },
  time_encode = function(dt) {
    torch_cos(dt$unsqueeze(-1L) * self$time_w + self$time_b)
  },
  update_memory = function(mem, last_t, ev) {
    # ev: previous batch (src, dst, t, edge_attr); "last" message aggregation per node.
    dev <- mem$device
    nodes <- c(ev$src, ev$dst)
    other <- c(ev$dst, ev$src)
    tt <- c(ev$t, ev$t)
    keep <- !duplicated(nodes, fromLast = TRUE)
    u <- nodes[keep]
    v <- other[keep]
    tu <- tt[keep]
    ui <- torch_tensor(u, dtype = torch_long(), device = dev)
    vi <- torch_tensor(v, dtype = torch_long(), device = dev)
    dt <- torch_tensor(tu - last_t[u], dtype = torch_float(), device = dev)
    parts <- list(mem$index_select(1L, ui), mem$index_select(1L, vi), self$time_encode(dt))
    if (self$edge_dim > 0L) {
      ea <- rbind(ev$edge_attr, ev$edge_attr)[keep, , drop = FALSE]
      parts[[4L]] <- torch_tensor(ea, dtype = torch_float(), device = dev)
    }
    s_new <- self$gru(torch_cat(parts, dim = 2L), mem$index_select(1L, ui))
    last_t[u] <- tu
    list(mem = mem$index_copy(1L, ui, s_new), last_t = last_t)
  },
  gru = function(x, h) {
    # Standard GRU cell (Cho et al. 2014), gates in the order reset, update, new.
    gx <- self$gru_x(x)$chunk(3L, dim = 2L)
    gh <- self$gru_h(h)$chunk(3L, dim = 2L)
    r <- torch_sigmoid(gx[[1L]] + gh[[1L]])
    z <- torch_sigmoid(gx[[2L]] + gh[[2L]])
    n <- torch_tanh(gx[[3L]] + r * gh[[3L]])
    (1 - z) * n + z * h
  },
  embed = function(mem, last_t, nodes, t, nbrs = NULL) {
    dev <- mem$device
    ni <- torch_tensor(nodes, dtype = torch_long(), device = dev)
    dt <- torch_tensor(t - last_t[nodes], dtype = torch_float(), device = dev)
    if (self$embedding == "graph_attention") {
      z <- self$attend(mem, ni, dt, t, nbrs)
    } else {
      z <- self$emb_self(torch_cat(list(mem$index_select(1L, ni), self$time_encode(dt)), dim = 2L))
    }
    if (self$embedding == "graph_sum" && !is.null(nbrs) && length(nbrs$query)) {
      qi <- torch_tensor(nbrs$query, dtype = torch_long(), device = dev)
      ji <- torch_tensor(nbrs$neighbor, dtype = torch_long(), device = dev)
      dtn <- torch_tensor(t[nbrs$query] - nbrs$t, dtype = torch_float(), device = dev)
      h <- self$emb_nbr(torch_cat(list(mem$index_select(1L, ji), self$time_encode(dtn)), dim = 2L))
      z <- z + scatter_mean(h, qi, length(nodes))
    }
    if (!is.null(self$static)) z <- z + self$static(ni)
    z
  },
  attend = function(mem, ni, dt, t, nbrs) {
    dev <- mem$device
    m <- ni$size(1L)
    H <- self$heads
    C <- self$att_q$out_features %/% H
    s_q <- mem$index_select(1L, ni)
    q <- self$att_q(torch_cat(list(s_q, self$time_encode(torch_zeros(m, device = dev))), dim = 2L))
    agg <- torch_zeros(m, H * C, device = dev)
    self$last_attention <- NULL
    if (!is.null(nbrs) && length(nbrs$query)) {
      qi <- torch_tensor(nbrs$query, dtype = torch_long(), device = dev)
      ji <- torch_tensor(nbrs$neighbor, dtype = torch_long(), device = dev)
      dtn <- torch_tensor(t[nbrs$query] - nbrs$t, dtype = torch_float(), device = dev)
      parts <- list(mem$index_select(1L, ji), self$time_encode(dtn))
      if (self$edge_dim > 0L) parts[[3L]] <- torch_tensor(nbrs$edge_attr, dtype = torch_float(), device = dev)
      kv <- torch_cat(parts, dim = 2L)
      k <- self$att_k(kv)$view(c(-1L, H, C))
      v <- self$att_v(kv)$view(c(-1L, H, C))
      a <- (q$view(c(-1L, H, C))$index_select(1L, qi) * k)$sum(dim = -1L) / sqrt(C)
      a <- scatter_softmax(a, qi, m)
      self$last_attention <- list(query = nbrs$query, neighbor = nbrs$neighbor, alpha = a$detach())
      agg <- scatter_sum((v * a$unsqueeze(-1L))$reshape(c(-1L, H * C)), qi, m)
    }
    self$att_out(torch_cat(list(agg, s_q, self$time_encode(dt)), dim = 2L))
  },
  score = function(z_src, z_dst) {
    self$decoder(torch_cat(list(z_src, z_dst), dim = 2L))$squeeze(-1L)
  }
)

# Most recent partners per node, kept in R as events arrive.
tgn_neighbor_store <- function(n, k, edge_dim = 0L) {
  list(nb = matrix(0L, n, k), t = matrix(NA_real_, n, k), k = k,
       e = if (edge_dim > 0L) array(0, c(n, k, edge_dim)))
}

tgn_store_add <- function(store, src, dst, t, edge_attr = NULL) {
  k <- store$k
  for (i in seq_along(src)) {
    for (pr in list(c(src[i], dst[i]), c(dst[i], src[i]))) {
      a <- pr[1L]
      store$nb[a, ] <- c(pr[2L], store$nb[a, ])[seq_len(k)]
      store$t[a, ] <- c(t[i], store$t[a, ])[seq_len(k)]
      if (!is.null(store$e)) {
        d <- dim(store$e)[3L]
        cur <- matrix(store$e[a, , ], k, d)
        store$e[a, , ] <- rbind(edge_attr[i, ], cur)[seq_len(k), , drop = FALSE]
      }
    }
  }
  store
}

tgn_store_query <- function(store, nodes) {
  nb <- store$nb[nodes, , drop = FALSE]
  tm <- store$t[nodes, , drop = FALSE]
  ok <- which(nb > 0L, arr.ind = TRUE)
  out <- list(query = ok[, 1L], neighbor = nb[ok], t = tm[ok])
  if (!is.null(store$e)) {
    n <- nrow(store$nb)
    flat <- matrix(store$e, n * store$k, dim(store$e)[3L])
    out$edge_attr <- flat[(ok[, 2L] - 1L) * n + nodes[ok[, 1L]], , drop = FALSE]
  }
  out
}

#' @rdname tgn_model
#' @param model A `tgn_model`.
#' @param split Result of [split_temporal()] (`train`, `val`, `test`).
#' @param epochs Passes over the training events.
#' @param batch_size Events per batch.
#' @param lr Learning rate.
#' @param seed Optional seed for weights, negatives and the epoch order.
#' @param device Device.
#' @return `train_tgn()`: a `temporal_fit` with `history`, `val_metrics` and
#'   `test_metrics` (ROC AUC and average precision against one random
#'   destination per event). Evaluation continues the memory built over the
#'   training (and, for test, validation) events, without gradient updates.
#' @export
train_tgn <- function(model, split, epochs = 10L, batch_size = 200L, lr = 1e-3, seed = NULL,
                      device = NULL) {
  if (!inherits(model, "tgn_model")) stopf("`model` must come from tgn_model().")
  for (nm in c("train", "val", "test")) {
    if (!inherits(split[[nm]], "temporal_graph_data")) stopf("`split` must come from split_temporal().")
  }
  if (model$edge_dim > 0L && is.null(split$train$edge_attr)) stopf("The model expects event features.")
  epochs <- check_count(epochs, "epochs", 1L)
  batch_size <- check_count(batch_size, "batch_size", 1L)
  dev <- resolve_device(device)
  if (!is.null(seed)) torch_manual_seed(seed)
  model$to(device = dev)
  n <- model$num_nodes
  opt <- optim_adam(model$parameters, lr = lr)

  run <- function(tg, state, train, neg_seed) {
    m <- length(tg$t)
    starts <- seq(1L, max(m, 1L), by = batch_size)
    negs <- with_optional_seed(neg_seed, sample.int(n, m, replace = TRUE))
    scores <- list()
    losses <- numeric(0)
    for (b in starts) {
      if (m == 0L) break
      idx <- b:min(m, b + batch_size - 1L)
      ev <- list(src = tg$src[idx], dst = tg$dst[idx], t = tg$t[idx],
                 edge_attr = if (model$edge_dim > 0L) tg$edge_attr[idx, , drop = FALSE])
      step <- function() {
        cur <- if (is.null(state$pending)) list(mem = state$mem, last_t = state$last_t)
               else model$update_memory(state$mem, state$last_t, state$pending)
        q <- c(ev$src, ev$dst, negs[idx])
        tq <- rep(ev$t, 3L)
        nb <- if (model$embedding != "time") tgn_store_query(state$store, q)
        z <- model$embed(cur$mem, cur$last_t, q, tq, nb)
        k <- length(idx)
        zs <- z[1:k, ]
        pos <- model$score(zs, z[(k + 1L):(2L * k), ])
        neg <- model$score(zs, z[(2L * k + 1L):(3L * k), ])
        list(cur = cur, pos = pos, neg = neg)
      }
      if (train) {
        opt$zero_grad()
        r <- step()
        loss <- nnf_binary_cross_entropy_with_logits(
          torch_cat(list(r$pos, r$neg)), torch_cat(list(torch_ones_like(r$pos), torch_zeros_like(r$neg))))
        loss$backward()
        opt$step()
        losses <- c(losses, as.numeric(loss$item()))
      } else {
        r <- with_no_grad(step())
        scores[[length(scores) + 1L]] <- cbind(as.numeric(as_r_array(r$pos)), as.numeric(as_r_array(r$neg)))
      }
      state$mem <- r$cur$mem$detach()
      state$last_t <- r$cur$last_t
      state$pending <- ev
      if (model$embedding != "time") {
        state$store <- tgn_store_add(state$store, ev$src, ev$dst, ev$t, ev$edge_attr)
      }
    }
    list(state = state, loss = if (length(losses)) mean(losses) else NA_real_,
         scores = if (length(scores)) do.call(rbind, scores))
  }
  fresh <- function() list(mem = torch_zeros(n, model$memory_dim, device = dev),
                           last_t = rep(min(split$train$t, 0), n), pending = NULL,
                           store = tgn_neighbor_store(n, model$num_neighbors, model$edge_dim))
  metrics <- function(sc) {
    if (is.null(sc)) return(c(roc_auc = NA_real_, average_precision = NA_real_))
    y <- c(rep(TRUE, nrow(sc)), rep(FALSE, nrow(sc)))
    c(roc_auc = metric_roc_auc(y, c(sc[, 1L], sc[, 2L])),
      average_precision = metric_average_precision(y, c(sc[, 1L], sc[, 2L])))
  }
  history <- data.frame(epoch = integer(), train_loss = numeric(), val_auc = numeric())
  for (ep in seq_len(epochs)) {
    model$train()
    tr <- run(split$train, fresh(), TRUE, if (is.null(seed)) NULL else seed + ep)
    model$eval()
    va <- run(split$val, tr$state, FALSE, if (is.null(seed)) NULL else seed + 104729L)
    history[ep, ] <- list(ep, tr$loss, metrics(va$scores)[["roc_auc"]])
  }
  te <- run(split$test, va$state, FALSE, if (is.null(seed)) NULL else seed + 130363L)
  structure(list(model = model, history = history, val_metrics = metrics(va$scores),
                 test_metrics = metrics(te$scores), task = "continuous time link prediction",
                 device = dev, seed = seed),
            class = "temporal_fit")
}
