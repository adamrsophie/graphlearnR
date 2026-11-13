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
