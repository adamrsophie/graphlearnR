make_three <- function() {
  list(
    graph_data(x = matrix(1:2, 2, 1), edge_index = rbind(1, 2), graph_y = 0, node_ids = c("a", "b")),
    graph_data(x = matrix(3:5, 3, 1), edge_index = rbind(c(1, 2, 3), c(2, 3, 1)), graph_y = 1),
    graph_data(x = matrix(6, 1, 1), graph_y = 2)
  )
}

test_that("batching offsets edges and records membership", {
  b <- graph_batch(make_three())
  expect_s3_class(b, "graph_batch")
  expect_equal(b$num_graphs, 3L)
  expect_equal(num_nodes(b), 6L)
  expect_equal(b$batch, c(1L, 1L, 2L, 2L, 2L, 3L))
  expect_equal(b$node_ptr, c(0L, 2L, 5L, 6L))
  expect_equal(b$edge_ptr, c(0L, 1L, 4L, 4L))
  expect_equal(b$edge_index, rbind(c(1L, 3L, 4L, 5L), c(2L, 4L, 5L, 3L)))
  expect_equal(b$graph_y, c(0, 1, 2))
  expect_equal(b$x[, 1], c(1, 2, 3, 4, 5, 6))
})

test_that("no edge connects two graphs of a batch", {
  gs <- lapply(1:5, function(s) random_graph(n = 4 + s, m = 10, seed = s))
  b <- graph_batch(gs)
  expect_equal(b$batch[b$edge_index[1, ]], b$batch[b$edge_index[2, ]])
})

test_that("unbatch reconstructs the input graphs", {
  gs <- c(make_three(), list(dataset_karate()))
  gs[[4]]$graph_y <- 3
  b <- graph_batch(gs)
  back <- unbatch(b)
  expect_length(back, 4)
  for (k in seq_along(gs)) {
    for (f in c("x", "edge_index", "num_nodes", "edge_pair", "node_ids")) {
      expect_equal(back[[k]][[f]], gs[[k]][[f]], info = paste(k, f))
    }
    expect_equal(back[[k]]$graph_y, gs[[k]]$graph_y)
  }
})

test_that("batching checks compatibility", {
  expect_error(graph_batch(list(graph_data(x = diag(2)), graph_data(x = diag(3)))), "node features")
  expect_error(graph_batch(list(graph_data(x = diag(2)),
                                graph_data(x = diag(2), edge_index = rbind(1, 2), directed = FALSE))),
               "directed and undirected")
  expect_error(graph_batch(list(graph_data(x = diag(2), graph_y = 1), graph_data(x = diag(2)))),
               "graph_y")
  expect_error(graph_batch(list()), "non-empty")
})

test_that("batching factor labels keeps levels; nested batches flatten", {
  gs <- dataset_synthetic_graphs(4, seed = 2)
  b <- graph_batch(gs)
  expect_s3_class(b$graph_y, "factor")
  expect_equal(levels(b$graph_y), c("cycle", "wheel"))
  bb <- graph_batch(list(graph_batch(gs[1:2]), gs[[3]]))
  expect_equal(bb$num_graphs, 3L)
})

test_that("global pooling over a batch matches per graph calculations", {
  skip_if_no_torch()
  gs <- lapply(1:4, function(s) random_graph(n = 3 + s, m = 6, f = 2, seed = s))
  b <- graph_batch(gs)
  x <- torch::torch_tensor(b$x)
  sums <- t(vapply(gs, function(g) colSums(g$x), numeric(2)))
  means <- t(vapply(gs, function(g) colMeans(g$x), numeric(2)))
  maxs <- t(vapply(gs, function(g) apply(g$x, 2, max), numeric(2)))
  expect_equal(as_mat(global_add_pool(x, b$batch)), sums, tolerance = 1e-5)
  expect_equal(as_mat(global_mean_pool(x, b$batch)), means, tolerance = 1e-5)
  expect_equal(as_mat(global_max_pool(x, b$batch)), maxs, tolerance = 1e-5)
})
