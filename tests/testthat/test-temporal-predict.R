test_that("TGN predictions continue from the stored state and refuse the past", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 400, seed = 2)
  sp <- split_temporal(tg)
  torch::torch_manual_seed(1)
  fit <- train_tgn(tgn_model(tg$num_nodes, 8, 4, 8, embedding = "graph_sum"), sp, epochs = 1,
                   batch_size = 50, seed = 1)
  last <- fit$last_time
  expect_equal(last, max(tg$t))
  pr <- rbind(c(1, 2, 3), c(4, 5, 6))
  p1 <- predict(fit, pairs = pr, time = last + 1)
  expect_length(p1, 3L)
  expect_true(all(p1 > 0 & p1 < 1))
  expect_equal(predict(fit, pairs = pr, time = last + 1), p1)
  expect_equal(qlogis(p1), predict(fit, pairs = pr, time = last + 1, type = "score"), tolerance = 1e-6)
  new <- temporal_graph_data(src = c(1, 2), dst = c(4, 5), t = c(last + 1, last + 2),
                             num_nodes = tg$num_nodes)
  p2 <- predict(fit, new_data = new, pairs = pr, time = last + 3)
  expect_false(isTRUE(all.equal(p1, p2)))
  # the stored state is not modified by a prediction
  expect_equal(predict(fit, pairs = pr, time = last + 1), p1)
  expect_error(predict(fit, pairs = pr, time = last - 1), "precede")
  old <- temporal_graph_data(src = 1, dst = 2, t = last - 5, num_nodes = tg$num_nodes)
  expect_error(predict(fit, new_data = old, pairs = pr, time = last), "precede")
})

test_that("dynamic link predictions use only earlier snapshots", {
  skip_if_no_torch()
  snaps <- temporal_snapshots(dataset_synthetic_temporal(seed = 1), breaks = 6)
  fit <- train_dynamic_link_model(snaps, hidden_dim = 8, epochs = 2, seed = 1)
  pr <- rbind(c(1, 2), c(3, 4))
  a <- predict(fit, snaps, pairs = pr, step = 4)
  later <- snaps
  later$graphs[[5]] <- later$graphs[[1]]
  later$graphs[[4]]$edge_index <- later$graphs[[4]]$edge_index[, 1, drop = FALSE]
  later$graphs[[4]]$edge_pair <- NULL
  expect_equal(predict(fit, later, pairs = pr, step = 4), a)
  expect_error(predict(fit, snaps, pairs = pr, step = 1), "step")
})

test_that("spatiotemporal forecasts match the fitted test windows", {
  skip_if_no_torch()
  sig <- dataset_synthetic_signal(seed = 1)
  fit <- train_spatiotemporal_model(sig, lags = 3, horizon = 1, hidden_dim = 8, epochs = 2, seed = 1)
  ends <- fit$windows$test
  p <- predict(fit, sig, ends = ends)
  expect_equal(dim(p), c(length(ends), sig$graph$num_nodes))
  truth <- sig$targets[ends + 1, ]
  expect_equal(fit$test_metrics[["mse"]], metric_mse(as.vector(truth), as.vector(p)), tolerance = 1e-5)
  # changing features after an origin does not change its forecast
  sig2 <- sig
  e <- ends[1]
  sig2$features[(e + 1):dim(sig$features)[1], , ] <- 0
  expect_equal(predict(fit, sig2, ends = e), p[1, , drop = FALSE])
})

test_that("temporal fits survive save and load with their state", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 300, seed = 3)
  torch::torch_manual_seed(2)
  fit <- train_tgn(tgn_model(tg$num_nodes, 8, 4, 8, embedding = "graph_attention"), split_temporal(tg),
                   epochs = 1, batch_size = 50, seed = 2)
  path <- withr::local_tempfile(fileext = ".rds")
  save_graph_fit(fit, path)
  back <- load_graph_fit(path)
  pr <- rbind(c(1, 5), c(2, 6))
  expect_equal(predict(back, pairs = pr, time = fit$last_time + 1),
               predict(fit, pairs = pr, time = fit$last_time + 1), tolerance = 1e-6)
})
