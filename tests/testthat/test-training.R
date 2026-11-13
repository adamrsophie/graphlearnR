test_that("node classification learns an SBM well above chance", {
  skip_if_no_torch()
  testthat::skip_on_cran()  # trains for many epochs
  g <- split_nodes(dataset_sbm(c(40, 40, 40), p_in = 0.15, p_out = 0.01, feature_signal = 0.5,
                               seed = 1), seed = 1)
  fit <- train_graph_model(gnn_model("sage", 8, 16, 3, dropout = 0.2), g, "node_classification",
                           epochs = 60, lr = 0.02, seed = 1)
  ev <- evaluate_graph_model(fit, g)
  expect_gt(ev$value[ev$metric == "accuracy"], 0.8)
  expect_s3_class(fit, "graph_fit")
  expect_equal(nrow(fit$history), 60)
  expect_output(print(fit), "node_classification")
  p <- predict(fit, g)
  expect_s3_class(p, "graph_predictions")
  expect_equal(nrow(p), 120)
  expect_equal(levels(p$.pred_class), levels(g$y))
  pr <- predict(fit, g, type = "prob")
  expect_equal(rowSums(pr), rep(1, 120), tolerance = 1e-5)
})

test_that("node regression and alternative losses run", {
  skip_if_no_torch()
  g <- dataset_sbm(c(20, 20), seed = 2)
  g$y <- as.numeric(g$y == "block2") + g$x[, 1] * 0.1
  g <- split_nodes(g, stratify = FALSE, seed = 2)
  for (l in c("mse", "l1", "huber")) {
    fit <- train_graph_model(gnn_model("gcn", 8, 8, 1), g, "node_regression", epochs = 15,
                             loss = l, seed = 1)
    expect_true(is.finite(tail(fit$history$train_loss, 1)), info = l)
  }
  ev <- evaluate_graph_model(fit, g)
  expect_setequal(ev$metric, c("mse", "rmse", "mae", "r2"))
  expect_equal(ncol(predict(fit, g)), 3)
})

test_that("training with the same seed is reproducible and leaves R's RNG alone", {
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), seed = 1)
  set.seed(99)
  before <- .Random.seed
  run <- function() {
    m <- {
      torch::torch_manual_seed(7)
      gnn_model("sage", 34, 8, 2)
    }
    train_graph_model(m, g, "node_classification", epochs = 10, seed = 7,
                      fanouts = c(4, 4), batch_size = 8)$history$train_loss
  }
  a <- run()
  b <- run()
  expect_equal(a, b)
  expect_identical(.Random.seed, before)
})

test_that("mini-batch training uses seed nodes only", {
  skip_if_no_torch()
  g <- split_nodes(dataset_sbm(c(30, 30), seed = 3), seed = 3)
  fit <- train_graph_model(gnn_model("sage", 8, 16, 2), g, "node_classification", epochs = 15,
                           fanouts = c(5, 5), batch_size = 16, seed = 3)
  expect_true(all(is.finite(fit$history$train_loss)))
  expect_gt(tail(fit$history$val_metric, 1), 0.6)
})

test_that("graph classification learns structural classes", {
  skip_if_no_torch()
  testthat::skip_on_cran()  # trains for many epochs
  gs <- dataset_synthetic_graphs(80, seed = 4)
  sp <- split_graphs(80, labels = vapply(gs, function(g) as.character(g$graph_y), ""), seed = 4)
  fit <- train_graph_model(gnn_model("gin", 1, 16, 2, pool = "sum", dropout = 0), gs[sp$train],
                           "graph_classification", val_data = gs[sp$val], epochs = 40,
                           batch_size = 16, seed = 4)
  ev <- evaluate_graph_model(fit, gs[sp$test])
  expect_gt(ev$value[ev$metric == "accuracy"], 0.85)
  p <- predict(fit, gs[sp$test])
  expect_equal(nrow(p), length(sp$test))
})

test_that("graph regression works with a dataset object", {
  skip_if_no_torch()
  ds <- graph_dataset(dataset_synthetic_graphs(30, task = "regression", seed = 5))
  fit <- train_graph_model(gnn_model("gin", 1, 8, 1, pool = "mean", dropout = 0), ds,
                           "graph_regression", val_data = ds[1:5], epochs = 20, seed = 5)
  ev <- evaluate_graph_model(fit, ds)
  expect_true("rmse" %in% ev$metric)
})

test_that("link prediction trains on training edges and evaluates held out pairs", {
  skip_if_no_torch()
  testthat::skip_on_cran()  # trains for many epochs
  g <- dataset_sbm(c(40, 40), p_in = 0.25, p_out = 0.01, seed = 6)
  sp <- split_edges(g, val = 0.1, test = 0.1, seed = 6)
  fit <- train_graph_model(gnn_model("gcn", 8, 16, dropout = 0), sp, "link_prediction",
                           epochs = 60, lr = 0.01, seed = 6)
  ev <- evaluate_graph_model(fit)
  expect_gt(ev$value[ev$metric == "roc_auc"], 0.7)
  expect_true(all(c("average_precision", "mrr", "hits@10") %in% ev$metric))
  s <- predict(fit, pairs = sp$test_pos, type = "prob")
  expect_true(all(s >= 0 & s <= 1))
  expect_equal(nrow(predict(fit, pairs = sp$test_pos)), ncol(sp$test_pos))
})

test_that("edge classification uses edge masks", {
  skip_if_no_torch()
  testthat::skip_on_cran()  # trains for many epochs
  g <- dataset_sbm(c(20, 20), p_in = 0.3, p_out = 0.05, seed = 7)
  same <- g$y[g$edge_index[1, ]] == g$y[g$edge_index[2, ]]
  g$edge_y <- factor(ifelse(same, "within", "between"))
  g <- split_edge_labels(g, seed = 7)
  fit <- train_graph_model(gnn_model("sage", 8, 16, dropout = 0), g, "edge_classification",
                           epochs = 40, seed = 7)
  ev <- evaluate_graph_model(fit, g)
  expect_gt(ev$value[ev$metric == "accuracy"], 0.7)
})

test_that("early stopping, schedulers and checkpoints work", {
  skip_if_no_torch()
  testthat::skip_on_cran()  # trains for many epochs
  g <- split_nodes(dataset_karate(), seed = 1)
  ck <- withr::local_tempfile(fileext = ".pt")
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification", epochs = 200,
                           lr = 0.05, early_stopping = 5, checkpoint_path = ck,
                           scheduler = list(type = "step", step_size = 5, gamma = 0.5), seed = 1)
  expect_lt(nrow(fit$history), 200)
  expect_true(file.exists(ck))
  expect_lt(tail(fit$history$lr, 1), 0.05)
  sd <- torch::torch_load(ck)
  expect_true(length(sd) > 0)
  for (s in list(list(type = "plateau", patience = 2), list(type = "cosine"))) {
    f2 <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification", epochs = 6,
                            scheduler = s, optimizer = "adamw", seed = 1)
    expect_equal(nrow(f2$history), 6)
  }
  f3 <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification", epochs = 3,
                          optimizer = function(p) torch::optim_sgd(p, lr = 0.1), grad_clip = 1,
                          class_weights = c(1, 2), seed = 1)
  expect_equal(f3$reproducibility$optimizer, "custom")
})

test_that("saved fits reload and predict identically", {
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gat", 34, 8, 2, heads = 2), g, "node_classification",
                           epochs = 10, seed = 1)
  path <- withr::local_tempfile(fileext = ".rds")
  save_graph_fit(fit, path)
  back <- load_graph_fit(path)
  expect_equal(predict(back, g, type = "prob"), predict(fit, g, type = "prob"), tolerance = 1e-6)
  expect_equal(back$levels, fit$levels)
  other <- withr::local_tempfile(fileext = ".rds")
  saveRDS(1, other)
  expect_error(load_graph_fit(other), "not a saved")
  sp <- split_edges(dataset_karate(), seed = 1)
  lf <- train_graph_model(gnn_model("gcn", 34, 8), sp, "link_prediction", epochs = 5, seed = 1)
  save_graph_fit(lf, path)
  lb <- load_graph_fit(path)
  expect_equal(predict(lb, sp$train_graph, type = "score", pairs = sp$test_pos),
               predict(lf, sp$train_graph, type = "score", pairs = sp$test_pos), tolerance = 1e-6)
})

test_that("reproducibility record captures settings", {
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("sage", 34, 8, 2), g, "node_classification", epochs = 2, seed = 3)
  r <- fit$reproducibility
  expect_equal(r$seed, 3)
  expect_equal(r$model_config$architecture, "sage")
  expect_equal(unname(r$splits$counts), c(sum(g$train_mask), sum(g$val_mask), sum(g$test_mask)))
  expect_equal(r$splits$members$test, which(g$test_mask))
  expect_true(all(c("R", "graphlearnR", "torch") %in% names(r$versions)))
  expect_equal(r$device, "cpu")
  expect_equal(r$precision, "Float")
  sp <- split_edges(dataset_karate(), seed = 1)
  f2 <- train_graph_model(gnn_model("gcn", 34, 8), sp, "link_prediction", epochs = 1,
                          decoder = "mlp", grad_clip = 1, seed = 1)
  r2 <- f2$reproducibility
  expect_equal(r2$splits$members$test_pos, sp$test_pos)
  expect_equal(r2$decoder$type, "mlp")
  expect_equal(r2$hyperparameters$grad_clip, 1)
})

test_that("training input errors are informative", {
  skip_if_no_torch()
  g <- dataset_karate()
  m <- gnn_model("sage", 34, 8, 2)
  expect_error(train_graph_model(m, g, "node_classification"), "split_nodes")
  expect_error(train_graph_model(m, g, "link_prediction"), "link_split")
  expect_error(train_graph_model(m, g, "edge_classification"), "edge_y")
  expect_error(train_graph_model(m, g, "graph_classification"), "list of graphs")
  expect_error(train_graph_model(list(), g), "nn_module")
})

test_that("classification accepts 0 based integer and numeric labels", {
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), seed = 1)
  g$y <- as.integer(g$y) - 1L
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification",
                           epochs = 5, seed = 1)
  expect_equal(fit$levels, c("0", "1"))
  ev <- evaluate_graph_model(fit, g)
  expect_true(all(is.finite(ev$value)))
  gs <- lapply(1:6, function(i) {
    graph_data(x = matrix(i %% 2, 3, 1), edge_index = rbind(1:2, 2:3), graph_y = i %% 2)
  })
  gf <- train_graph_model(gnn_model("gin", 1, 4, 2, pool = "mean"), gs,
                          "graph_classification", epochs = 2, batch_size = 2, seed = 1)
  expect_equal(gf$levels, c("0", "1"))
  expect_equal(dim(predict(gf, gs, type = "embedding")), c(6L, 4L))
})

test_that("multi target regression is evaluated column by column", {
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), stratify = FALSE, seed = 1)
  g$y <- cbind(g$x[, 1], g$x[, 2] * 100)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_regression",
                           epochs = 3, seed = 1)
  p <- predict(fit, g, type = "response")
  ev <- evaluate_graph_model(fit, g)
  m <- g$test_mask
  expect_equal(ev$value[ev$metric == "mse"],
               mean((g$y[m, , drop = FALSE] - p[m, , drop = FALSE])^2), tolerance = 1e-6)
})

test_that("checkpoints keep the decoder and verbose = 0 is silent", {
  skip_if_no_torch()
  sp <- split_edges(dataset_karate(), seed = 1)
  ck <- withr::local_tempfile(fileext = ".pt")
  fit <- train_graph_model(gnn_model("gcn", 34, 8), sp, "link_prediction", epochs = 3,
                           decoder = "mlp", checkpoint_path = ck, verbose = 0, seed = 1)
  expect_true(file.exists(paste0(ck, ".decoder")))
  dec <- link_decoder("mlp", 8)
  dec$load_state_dict(torch::torch_load(paste0(ck, ".decoder")))
  expect_silent(train_graph_model(gnn_model("gcn", 34, 8, 2), split_nodes(dataset_karate(), seed = 1),
                                  "node_classification", epochs = 2, verbose = 0, seed = 1))
  expect_warning(train_graph_model(gnn_model("gcn", 34, 8, 2), split_nodes(dataset_karate(), seed = 1),
                                   "node_classification", epochs = 1, loss = "focal",
                                   class_weights = c(1, 2), seed = 1), "cross entropy")
})

test_that("edge predictions take the features of the requested edges", {
  skip_if_no_torch()
  g <- dataset_karate()
  g$edge_attr <- matrix(seq_len(num_edges(g)) / num_edges(g), ncol = 1)
  g$edge_y <- factor(ifelse(g$edge_index[1, ] < g$edge_index[2, ], "up", "down"))
  g <- split_edge_labels(g, seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8), g, "edge_classification", epochs = 2, seed = 1)
  all <- predict(fit, g, type = "response")
  rev <- rev(seq_len(num_edges(g)))
  sub <- predict(fit, g, type = "response", pairs = g$edge_index[, rev])
  expect_equal(sub, all[rev, , drop = FALSE], tolerance = 1e-5)
  expect_error(predict(fit, g, pairs = rbind(1, 1)), "not edges")
})

test_that("saved heterogeneous link fits still predict", {
  skip_if_no_torch()
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
  sp <- split_edges_hetero(hg, "drug__targets__protein", seed = 1)
  m <- hetero_gnn_model(hg, hidden_dim = 8, num_layers = 1)
  fit <- train_graph_model(m, sp, "link_prediction", epochs = 2, seed = 1)
  path <- withr::local_tempfile(fileext = ".rds")
  save_graph_fit(fit, path)
  back <- load_graph_fit(path)
  s1 <- predict(fit, pairs = sp$test_pos, type = "score")
  s2 <- predict(back, sp$train_graph, pairs = sp$test_pos, type = "score")
  expect_equal(s1, s2, tolerance = 1e-5)
  expect_error(predict(back, pairs = sp$test_pos), "new_data")
})

test_that("a 2 by 2 pair matrix with from and to names is a list of pairs", {
  m <- cbind(from = c(1, 2), to = c(33, 34))
  expect_equal(normalize_edge_index(m), rbind(c(1L, 2L), c(33L, 34L)))
})

test_that("mini batch options that a task would ignore are errors", {
  skip_if_no_torch()
  sp <- split_edges(dataset_karate(), seed = 1)
  expect_error(train_graph_model(gnn_model("gcn", 34, 8), sp, "link_prediction", epochs = 1,
                                 batch_size = 8), "batch_size")
  hg <- dataset_synthetic_hetero(seed = 1)
  expect_error(train_graph_model(hetero_gnn_model(hg, hidden_dim = 8, output_dim = 3, target_type = "drug"), hg,
                                 "node_classification", target_type = "drug", epochs = 1,
                                 loader = cluster_loader(dataset_karate(), 2)), "loader")
})

test_that("write_graph_dataset refuses to mix with an existing dataset", {
  dir <- withr::local_tempdir()
  gs <- list(dataset_karate(), dataset_karate())
  write_graph_dataset(gs, dir)
  expect_error(write_graph_dataset(gs[1], dir), "overwrite")
  write_graph_dataset(gs[1], dir, overwrite = TRUE)
  expect_length(graph_dataset_disk(dir), 1L)
})

test_that("fits with a named mlp decoder survive save and load", {
  skip_if_no_torch()
  sp <- split_edges(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8), sp, "link_prediction", epochs = 3,
                           decoder = "mlp", early_stopping = 2, seed = 1)
  path <- withr::local_tempfile(fileext = ".rds")
  save_graph_fit(fit, path)
  back <- load_graph_fit(path)
  expect_equal(predict(back, sp$train_graph, pairs = sp$test_pos, type = "score"),
               predict(fit, pairs = sp$test_pos, type = "score"), tolerance = 1e-5)
})

test_that("filtered link evaluation ranks each positive against its own tails", {
  skip_if_no_torch()
  sp <- split_edges(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8), sp, "link_prediction", epochs = 5, seed = 1)
  ev <- evaluate_graph_model(fit, filtered_negatives = 5, ks = c(1, 6), seed = 1)
  expect_true(all(c("filtered_mrr", "filtered_hits@1", "filtered_hits@6") %in% ev$metric))
  # With 5 negatives a rank is at most 6.
  expect_equal(ev$value[ev$metric == "filtered_hits@6"], 1)
  v <- ev$value[ev$metric == "filtered_mrr"]
  expect_true(v > 0 && v <= 1)
})
