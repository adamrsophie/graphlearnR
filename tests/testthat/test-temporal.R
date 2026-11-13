test_that("temporal graphs sort events by time and validate them", {
  tg <- temporal_graph_data(src = c(3, 1, 2), dst = c(1, 2, 3), t = c(5, 1, 3),
                            edge_attr = matrix(c(30, 10, 20), 3, 1), y = c("c", "a", "b"))
  expect_equal(tg$t, c(1, 3, 5))
  expect_equal(tg$src, c(1L, 2L, 3L))
  expect_equal(tg$edge_attr[, 1], c(10, 20, 30))
  expect_equal(tg$y, c("a", "b", "c"))
  expect_equal(tg$event_id, c(2L, 3L, 1L))
  expect_equal(num_edges(tg), 3L)
  expect_error(temporal_graph_data(1, 2, NA), "finite")
  expect_error(temporal_graph_data(1, 5, 1, num_nodes = 3), "1:3")
  bad <- tg
  bad$t <- c(3, 1, 2)
  expect_error(validate_temporal_graph(bad), "time order")
  expect_output(print(tg), "3 events")
})

test_that("snapshots follow the stated interval rule", {
  tg <- temporal_graph_data(src = c(1, 2, 1, 3, 1), dst = c(2, 3, 2, 1, 3), t = c(0, 1, 1.5, 2, 4))
  s <- temporal_snapshots(tg, breaks = c(0, 2, 4))
  expect_length(s$graphs, 2)
  g1 <- s$graphs[[1]]
  expect_equal(g1$edge_index, rbind(c(1L, 2L), c(2L, 3L)))
  expect_equal(g1$edge_weight, c(2, 1))
  g2 <- s$graphs[[2]]
  expect_equal(ncol(g2$edge_index), 2L)
  cum <- temporal_snapshots(tg, breaks = c(0, 2, 4), cumulative = TRUE)
  expect_equal(sum(cum$graphs[[2]]$edge_weight), 5)
  expect_equal(s$start, c(0, 2))
  expect_output(print(s), "2 snapshots")
})

test_that("temporal neighbors never use events at or after the query time", {
  tg <- dataset_synthetic_temporal(30, 300, seed = 1)
  q_nodes <- 1:30
  q_times <- withr::with_seed(2, runif(30, 10, 90))
  nb <- temporal_neighbors(tg, q_nodes, q_times, k = 5, direction = "both")
  expect_true(all(nb$t < q_times[nb$query]))
  expect_true(all(table(nb$query) <= 5))
  for (q in unique(nb$query)[1:5]) {
    ev <- which((tg$src == q_nodes[q] | tg$dst == q_nodes[q]) & tg$t < q_times[q])
    expect_equal(max(nb$t[nb$query == q]), max(tg$t[ev]))
  }
  same <- temporal_neighbors(tg, tg$dst[50], tg$t[50], k = 100)
  expect_false(50 %in% same$event)
  incl <- temporal_neighbors(tg, tg$dst[50], tg$t[50], k = 100, strict = FALSE)
  expect_true(50 %in% incl$event)
})

test_that("temporal splits respect cut points", {
  tg <- dataset_synthetic_temporal(20, 200, seed = 3)
  sp <- split_temporal(tg, val_time = 40, test_time = 70)
  expect_true(all(sp$train$t < 40))
  expect_true(all(sp$val$t >= 40 & sp$val$t < 70))
  expect_true(all(sp$test$t >= 70))
  expect_equal(length(sp$train$t) + length(sp$val$t) + length(sp$test$t), 200)
  q <- split_temporal(tg)
  expect_lte(max(q$train$t), min(q$val$t))
  expect_error(split_temporal(tg, 50, 40), "precede")
})

test_that("recurrent outputs at a step do not depend on later inputs", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  g <- dataset_karate(features = "none")
  tt <- graph_tensors(g)
  m <- temporal_gnn_model(2, 8, 1)
  xs <- lapply(1:5, function(i) torch::torch_randn(34, 2))
  out1 <- m(xs, tt$edge_index)$outputs
  xs2 <- xs
  xs2[[5]] <- torch::torch_randn(34, 2) * 100
  xs2[[4]] <- torch::torch_randn(34, 2) * 100
  out2 <- m(xs2, tt$edge_index)$outputs
  for (s in 1:3) expect_equal(as_mat(out2[[s]]), as_mat(out1[[s]]))
  expect_false(isTRUE(all.equal(as_mat(out2[[5]]), as_mat(out1[[5]]))))
})

test_that("spatiotemporal forecasting trains and reports a naive baseline", {
  skip_if_no_torch()
  sig <- dataset_synthetic_signal(12, 40, seed = 1)
  expect_output(print(sig), "40 steps")
  fit <- train_spatiotemporal_model(sig, lags = 3, horizon = 1, hidden_dim = 8, epochs = 5, seed = 1)
  expect_s3_class(fit, "temporal_fit")
  expect_true(all(fit$windows$train < min(fit$windows$val)))
  expect_true(all(fit$windows$val < min(fit$windows$test)))
  expect_true(all(c("mse", "mae") %in% names(fit$test_metrics)))
  expect_output(print(fit), "baseline")
})

test_that("dynamic link prediction evaluates later snapshots only", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(30, 600, seed = 2)
  snaps <- temporal_snapshots(tg, breaks = 6)
  fit <- train_dynamic_link_model(snaps, hidden_dim = 8, epochs = 5, seed = 2)
  expect_true(min(fit$eval_steps) > max(fit$train_steps))
  expect_true(all(fit$step_metrics$step %in% fit$eval_steps))
  expect_true(all(fit$test_metrics >= 0 & fit$test_metrics <= 1))
  base <- dynamic_link_baseline(snaps, fit$eval_steps, seed = 2)
  expect_gt(base[["roc_auc"]], 0.5)
  expect_error(train_dynamic_link_model(snaps, train_steps = c(2, 6)), "after all training")
})

test_that("spatiotemporal_signal validates its array", {
  g <- dataset_karate(features = "none")
  expect_error(spatiotemporal_signal(g, array(0, c(3, 5, 1))), "N = 34")
  s <- spatiotemporal_signal(g, matrix(0, 3, 34))
  expect_equal(dim(s$features), c(3, 34, 1))
})
