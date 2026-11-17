# A two layer sum model: node output = sum over 2 hop paths of feature 1.
toy_sum_model <- torch::nn_module(
  initialize = function() self$w <- torch::nn_parameter(torch::torch_ones(1)),
  forward = function(x, edge_index, edge_weight = NULL, return_embedding = FALSE, ...) {
    h <- message_passing(x, edge_index, edge_weight = edge_weight)
    h <- message_passing(h, edge_index, edge_weight = edge_weight) + h
    if (return_embedding) return(h)
    torch::torch_cat(list(h[, 1:1] * self$w, -h[, 1:1] * self$w), dim = 2)
  }
)

star_graph <- function() {
  # node 1 linked to 2..6, node 6 linked to 7; only node 7 carries signal
  pairs <- rbind(c(2, 3, 4, 5, 6, 7), c(1, 1, 1, 1, 1, 6))
  x <- matrix(0, 7, 2)
  x[7, 1] <- 5
  x[, 2] <- 1
  graph_data(x = x, edge_index = pairs, num_nodes = 7, directed = FALSE)
}

test_that("subgraphx finds the path that carries the signal", {
  skip_if_no_torch()
  g <- star_graph()
  m <- toy_sum_model()
  ex <- subgraphx(m, g, target = 1, max_nodes = 3, rollouts = 15, samples = 20, seed = 1)
  expect_s3_class(ex, "graph_explanation")
  expect_setequal(ex$subgraph_nodes, c(1, 6, 7))
  expect_gt(ex$score, 0)
  inside <- ex$edges$attribution == 1
  expect_true(all(ex$edges$from[inside] %in% ex$subgraph_nodes))
  expect_lte(length(ex$subgraph_nodes), 3)
  # reproducible with a seed
  ex2 <- subgraphx(m, g, target = 1, max_nodes = 3, rollouts = 15, samples = 20, seed = 1)
  expect_identical(ex2$subgraph_nodes, ex$subgraph_nodes)
  expect_error(subgraphx(m, g, target = 99), "node of")
})

test_that("subgraphx explains graph level and attention models", {
  skip_if_no_torch()
  gs <- dataset_synthetic_graphs(6, seed = 1)
  torch::torch_manual_seed(1)
  m <- gnn_model("gat", ncol(gs[[1]]$x), 8, 2, pool = "mean")
  ex <- subgraphx(m, gs[[1]], max_nodes = 3, rollouts = 3, samples = 4, seed = 1)
  expect_lte(length(ex$subgraph_nodes), 3)
  expect_true(nrow(ex$candidates) >= 1)
})

test_that("pg_explainer learns masks that keep the signal path", {
  skip_if_no_torch()
  g <- star_graph()
  m <- toy_sum_model()
  pg <- pg_explainer(m, g, targets = 1, epochs = 100, lr = 0.02, edge_size = 0.1, seed = 1)
  expect_s3_class(pg, "pg_explainer")
  expect_lt(mean(utils::tail(pg$history, 5)), mean(utils::head(pg$history, 5)))
  ex <- predict(pg, g, target = 1)
  e <- ex$edges
  expect_true(all(e$attribution >= 0 & e$attribution <= 1))
  on_path <- (e$from == 7 & e$to == 6) | (e$from == 6 & e$to == 1)
  expect_gt(min(e$attribution[on_path]), max(e$attribution[!on_path]))
  expect_equal(predict(pg, g, target = 1)$edges, e)
  expect_error(predict(pg, g), "target")
})

test_that("pg_explainer works for graph models and refuses models without edge weights", {
  skip_if_no_torch()
  gs <- dataset_synthetic_graphs(6, seed = 1)
  # random features: with identical nodes the output can not depend on edges
  withr::local_seed(1)
  gs <- lapply(gs, function(g) { g$x <- matrix(rnorm(2 * num_nodes(g)), num_nodes(g), 2); g })
  torch::torch_manual_seed(1)
  m <- gnn_model("gcn", 2, 8, 2, pool = "mean")
  pg <- pg_explainer(m, gs, epochs = 2, seed = 1)
  ex <- predict(pg, gs[[2]])
  expect_equal(nrow(ex$edges), num_edges(gs[[2]]))
  ma <- gnn_model("gat", 2, 8, 2, pool = "mean")
  expect_error(pg_explainer(ma, gs, epochs = 1), "edge_weight")
  expect_error(pg_explainer(m, list(1, 2)), "graph_data")
})
