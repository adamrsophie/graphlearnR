test_that("tgn memory update uses only the most recent message per node", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  m <- tgn_model(5, memory_dim = 4, time_dim = 2, embedding_dim = 4)
  mem <- torch::torch_randn(5, 4)
  ev1 <- list(src = c(1L, 1L), dst = c(2L, 3L), t = c(1, 2))
  ev2 <- list(src = 1L, dst = 3L, t = 2)
  last <- rep(0, 5)
  a <- m$update_memory(mem, last, ev1)
  b <- m$update_memory(mem, last, ev2)
  # node 1's last event is (1, 3, t = 2) in both; node 2 only appears in ev1
  expect_equal(as_mat(a$mem[1, ]$unsqueeze(1)), as_mat(b$mem[1, ]$unsqueeze(1)), tolerance = 1e-6)
  expect_equal(a$last_t, c(2, 1, 2, 0, 0))
  expect_equal(as_mat(a$mem[4:5, ]), as_mat(mem[4:5, ]))
  expect_false(isTRUE(all.equal(as_mat(a$mem[2, ]$unsqueeze(1)), as_mat(mem[2, ]$unsqueeze(1)))))
})

test_that("tgn gru matches the GRU cell formula", {
  skip_if_no_torch()
  torch::torch_manual_seed(2)
  m <- tgn_model(3, memory_dim = 2, time_dim = 2, embedding_dim = 2)
  x <- torch::torch_randn(1, 6)
  h <- torch::torch_randn(1, 2)
  wx <- as_mat(m$gru_x$weight); bx <- as.numeric(as.array(m$gru_x$bias$detach()))
  wh <- as_mat(m$gru_h$weight); bh <- as.numeric(as.array(m$gru_h$bias$detach()))
  gx <- as.vector(wx %*% as.vector(as.array(x)) + bx)
  gh <- as.vector(wh %*% as.vector(as.array(h)) + bh)
  sg <- function(v) 1 / (1 + exp(-v))
  r <- sg(gx[1:2] + gh[1:2]); z <- sg(gx[3:4] + gh[3:4])
  n <- tanh(gx[5:6] + r * gh[5:6])
  ref <- (1 - z) * n + z * as.vector(as.array(h))
  expect_equal(as.vector(as.array(m$gru(x, h)$detach())), ref, tolerance = 1e-6)
})

test_that("train_tgn learns and later events never change earlier results", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 1500, seed = 1)
  sp <- split_temporal(tg)
  torch::torch_manual_seed(1)
  fit <- train_tgn(tgn_model(tg$num_nodes, 32, 8, 32), sp, epochs = 10, batch_size = 50,
                   lr = 0.01, seed = 1)
  expect_s3_class(fit, "temporal_fit")
  expect_gt(fit$test_metrics[["roc_auc"]], 0.7)
  expect_lt(tail(fit$history$train_loss, 1), fit$history$train_loss[1])
  # change every test event: training history and validation metrics stay identical
  sp2 <- sp
  sp2$test$dst <- rev(sp2$test$dst)
  run <- function(s) {
    torch::torch_manual_seed(3)
    train_tgn(tgn_model(tg$num_nodes, 8, 4, 8, embedding = "graph_sum"), s, epochs = 2,
              batch_size = 100, seed = 3)
  }
  f1 <- run(sp)
  f2 <- run(sp2)
  expect_equal(f1$history, f2$history)
  expect_equal(f1$val_metrics, f2$val_metrics)
  expect_error(train_tgn(list(), sp), "tgn_model")
  expect_error(train_tgn(tgn_model(5), list(train = 1)), "split_temporal")
})

test_that("tgn neighbor store keeps the most recent partners and their features", {
  st <- graphlearnR:::tgn_neighbor_store(4, 2, edge_dim = 2)
  ea <- rbind(c(1, 10), c(2, 20), c(3, 30))
  st <- graphlearnR:::tgn_store_add(st, c(1, 1, 2), c(2, 3, 3), c(1, 2, 3), ea)
  q <- graphlearnR:::tgn_store_query(st, c(1, 3, 4))
  # node 1: partners 3 (t = 2) then 2 (t = 1); node 3: partners 2 (t = 3) then 1 (t = 2)
  ref <- data.frame(query = c(1, 2, 1, 2), neighbor = c(3, 2, 2, 1), t = c(2, 3, 1, 2))
  got <- data.frame(query = q$query, neighbor = q$neighbor, t = q$t)
  o <- order(got$query, -got$t)
  r <- order(ref$query, -ref$t)
  expect_equal(unname(as.matrix(got[o, ])), unname(as.matrix(ref[r, ])))
  expect_equal(unname(q$edge_attr[o, ]), rbind(c(2, 20), c(1, 10), c(3, 30), c(2, 20)))
})

test_that("tgn graph attention matches an explicit computation", {
  skip_if_no_torch()
  torch::torch_manual_seed(4)
  m <- tgn_model(4, memory_dim = 3, time_dim = 2, embedding_dim = 4, edge_dim = 1,
                 embedding = "graph_attention", heads = 2, node_embedding = FALSE)
  m$eval()
  mem <- torch::torch_randn(4, 3)
  last <- c(0, 0.5, 1, 0)
  nb <- list(query = c(1L, 1L), neighbor = c(2L, 3L), t = c(1, 2), edge_attr = matrix(c(0.3, -0.7), 2, 1))
  z <- as_mat(m$embed(mem, last, c(1L, 4L), c(3, 3), nb))
  W <- function(l) list(w = as_mat(l$weight), b = as.numeric(as.array(l$bias$detach())))
  ap <- function(p, v) as.vector(p$w %*% v + p$b)
  te <- function(dt) cos(dt * as.numeric(as.array(m$time_w$detach())) + as.numeric(as.array(m$time_b$detach())))
  M <- as_mat(mem)
  q <- ap(W(m$att_q), c(M[1, ], te(0)))
  kv <- lapply(1:2, function(e) c(M[nb$neighbor[e], ], te(3 - nb$t[e]), nb$edge_attr[e, ]))
  k <- lapply(kv, function(v) ap(W(m$att_k), v))
  v <- lapply(kv, function(v) ap(W(m$att_v), v))
  agg <- unlist(lapply(1:2, function(h) {
    idx <- (2 * h - 1):(2 * h)
    s <- vapply(1:2, function(e) sum(q[idx] * k[[e]][idx]) / sqrt(2), 0)
    a <- exp(s) / sum(exp(s))
    a[1] * v[[1]][idx] + a[2] * v[[2]][idx]
  }))
  mlp <- function(x) {
    h <- pmax(ap(W(m$att_out[[1]]), x), 0)
    ap(W(m$att_out[[3]]), h)
  }
  expect_equal(z[1, ], mlp(c(agg, M[1, ], te(3))), tolerance = 1e-5)
  # node 4 has no partners: zero attention output
  expect_equal(z[2, ], mlp(c(rep(0, 4), M[4, ], te(3))), tolerance = 1e-5)
  s <- as_mat(scatter_sum(m$last_attention$alpha, torch::torch_tensor(m$last_attention$query), 1))
  expect_equal(unname(s), matrix(1, 1, 2), tolerance = 1e-6)
  expect_error(tgn_model(4, embedding_dim = 5, embedding = "graph_attention", heads = 2), "divisible")
})

test_that("train_tgn learns with graph attention and event features", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 1200, seed = 2)
  if (is.null(tg$edge_attr)) tg$edge_attr <- matrix(withr::with_seed(3, stats::rnorm(length(tg$t))), ncol = 1)
  sp <- split_temporal(tg)
  torch::torch_manual_seed(1)
  m <- tgn_model(tg$num_nodes, 16, 8, 16, edge_dim = ncol(tg$edge_attr), embedding = "graph_attention",
                 num_neighbors = 5)
  fit <- train_tgn(m, sp, epochs = 5, batch_size = 50, lr = 0.01, seed = 1)
  expect_gt(fit$test_metrics[["roc_auc"]], 0.6)
  expect_true(all(is.finite(fit$history$train_loss)))
})

test_that("tgn memory uses the latest event of a node in either role", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  m <- tgn_model(5, memory_dim = 4, time_dim = 2, embedding_dim = 4)
  mem <- torch::torch_randn(5, 4)
  a <- m$update_memory(mem, rep(0, 5), list(src = c(1L, 2L), dst = c(2L, 3L), t = c(1, 2)))
  b <- m$update_memory(mem, rep(0, 5), list(src = 2L, dst = 3L, t = 2))
  expect_equal(a$last_t, c(1, 2, 2, 0, 0))
  expect_equal(as_mat(a$mem[2, ]$unsqueeze(1)), as_mat(b$mem[2, ]$unsqueeze(1)), tolerance = 1e-6)
  later <- m$update_memory(mem, c(0, 5, 0, 0, 0), list(src = 2L, dst = 3L, t = 2))
  expect_equal(later$last_t[2], 5)
})

test_that("tgn negatives avoid the event's own nodes and batches keep timestamps whole", {
  neg <- tgn_negatives(4L, c(1L, 2L, 3L, 1L), c(2L, 3L, 4L, 4L), seed = 1)
  expect_true(all(neg != c(1L, 2L, 3L, 1L) & neg != c(2L, 3L, 4L, 4L)))
  b <- tgn_batches(c(1, 1, 2, 2, 2, 3), 2L)
  expect_equal(b, list(1:2, 3:5, 6L))
})

test_that("train_tgn rejects periods that overlap in time", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 300, seed = 1)
  sp <- split_temporal(tg)
  sp$val <- sp$train
  expect_error(train_tgn(tgn_model(tg$num_nodes, 8, 4, 8), sp, epochs = 1), "overlap")
})
