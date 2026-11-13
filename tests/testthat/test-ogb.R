test_that("read_ogb_node converts indices, labels and the official split", {
  g <- read_ogb_node(system.file("extdata", "ogbn-toy", package = "graphlearnR"))
  expect_equal(num_nodes(g), 6L)
  expect_true(g$directed)
  expect_equal(g$edge_index[, 6], c(6L, 1L))
  expect_equal(as.character(g$y), c("0", "1", "2", "0", "1", "2"))
  expect_equal(which(g$train_mask), 1:3)
  expect_equal(which(g$val_mask), 4L)
  expect_equal(which(g$test_mask), 5:6)
  expect_equal(g$metadata$split, "time")
  u <- read_ogb_node(system.file("extdata", "ogbn-toy", package = "graphlearnR"), add_inverse_edge = TRUE)
  expect_equal(num_edges(u), 12L)
  expect_error(read_ogb_node(tempdir()), "OGB node dataset")
})

test_that("read_ogb_graph splits edges per graph and keeps local indices", {
  ds <- read_ogb_graph(system.file("extdata", "ogbg-toy", package = "graphlearnR"))
  expect_length(ds, 2L)
  g1 <- ds[[1]]
  g2 <- ds[[2]]
  expect_equal(num_nodes(g1), 3L)
  expect_equal(num_edges(g1, TRUE), 2L)
  expect_equal(num_edges(g2), 2L)
  expect_equal(g2$edge_index, rbind(c(1L, 2L), c(2L, 1L)))
  expect_equal(g2$edge_attr[1, ], c(2, 0))
  expect_equal(g2$x[, 1], c(6, 5))
  expect_equal(as.character(c(g1$graph_y, g2$graph_y)), c("0", "1"))
  expect_equal(ds$metadata$split, list(train = 1L, val = 2L, test = integer(0)))
  expect_error(read_ogb_graph(tempdir()), "OGB graph dataset")
  expect_error(download_ogb("ogbl-collab"), "Unknown OGB dataset")
})

test_that("ogb_evaluate follows the OGB per task rules", {
  expect_equal(unname(ogb_evaluate("ogbg-molhiv", c(0, 1, 1, 0), c(0.1, 0.8, 0.6, 0.3))), 1)
  truth <- cbind(c(0, 1, NA, 1), c(1, 1, 1, NA), c(0, 1, 0, 1))
  pred <- cbind(c(0.2, 0.9, 0.5, 0.7), c(0.6, 0.1, 0.3, 0.2), c(0.9, 0.1, 0.8, 0.2))
  # task 2 has one class and is skipped; task 1 is perfect, task 3 reversed
  ap3 <- metric_average_precision(c(FALSE, TRUE, FALSE, TRUE), c(0.9, 0.1, 0.8, 0.2))
  expect_equal(unname(ogb_evaluate("ogbg-molpcba", truth, pred)), mean(c(1, ap3)))
  expect_equal(unname(ogb_evaluate("ogbn-arxiv", c(1, 2, 3), c(1, 2, 0))), 2 / 3)
})

test_that("bce_masked ignores missing multi task labels", {
  skip_if_no_torch()
  gs <- dataset_synthetic_graphs(30, seed = 1)
  for (i in seq_along(gs)) {
    yy <- c(as.numeric(gs[[i]]$graph_y == levels(gs[[i]]$graph_y)[1]), NA)
    gs[[i]]$graph_y <- matrix(if (i %% 2) yy else rev(yy), 1)
  }
  fit <- train_graph_model(gnn_model("gin", ncol(gs[[1]]$x), 8, 2, pool = "sum"), gs,
                           "graph_regression", loss = "bce_masked", epochs = 3, seed = 1)
  expect_true(all(is.finite(fit$history$train_loss)))
  p <- predict(fit, gs, type = "response")
  expect_equal(dim(p), c(30L, 2L))
})
