test_that("igraph round trip is lossless for supported fields (directed)", {
  g <- graph_data(x = matrix(c(1, 2, 3, 4, 5, 6), 3, 2), edge_index = rbind(c(1, 2, 3), c(2, 3, 1)),
                  y = factor(c("a", "b", "a")), edge_weight = c(0.1, 0.2, 0.3),
                  edge_attr = matrix(c(7, 8, 9), 3, 1), node_ids = c("u", "v", "w"),
                  train_mask = c(TRUE, FALSE, FALSE))
  g2 <- from_igraph(to_igraph(g))
  for (f in c("x", "edge_index", "edge_weight", "num_nodes", "directed", "train_mask")) {
    expect_equal(unname(g2[[f]]), unname(g[[f]]), info = f)
  }
  expect_equal(unname(g2$edge_attr), unname(g$edge_attr))
  expect_equal(as.character(g2$y), as.character(g$y))
  expect_equal(g2$node_ids, g$node_ids)
})

test_that("igraph round trip keeps undirected graphs and pair structure", {
  g <- dataset_karate()
  ig <- to_igraph(g)
  expect_false(igraph::is_directed(ig))
  expect_equal(igraph::ecount(ig), 78)
  g2 <- from_igraph(ig)
  expect_equal(g2$edge_index, g$edge_index)
  expect_equal(g2$edge_pair, g$edge_pair)
  expect_equal(unname(g2$x), unname(g$x))
})

test_that("from_igraph reads names and chosen attributes", {
  ig <- igraph::make_graph(~ A - B, B - C)
  igraph::V(ig)$size <- c(1, 2, 3)
  igraph::E(ig)$weight <- c(5, 6)
  g <- from_igraph(ig, x_attrs = "size")
  expect_equal(g$node_ids, c("A", "B", "C"))
  expect_equal(g$x[, 1], c(1, 2, 3))
  expect_equal(g$edge_weight, c(5, 5, 6, 6))
  expect_error(from_igraph(ig, x_attrs = "nope"), "not found")
})

test_that("edge lists map nonconsecutive identifiers", {
  edges <- data.frame(src = c(10, 30, 20), dst = c(30, 20, 99), w = 1:3)
  nodes <- data.frame(id = c(99, 10, 20, 30, 50), f = c(1, 2, 3, 4, 5), lab = letters[1:5])
  g <- graph_from_edgelist(edges, "src", "dst", nodes = nodes, feature_cols = "f",
                           label_col = "lab", weight_col = "w")
  expect_equal(num_nodes(g), 5L)
  expect_equal(g$node_ids, nodes$id)
  expect_equal(g$edge_index, rbind(c(2L, 4L, 3L), c(4L, 3L, 1L)))
  expect_equal(g$x[, 1], nodes$f)
  el <- edgelist_from_graph(g)
  expect_equal(el$from, edges$src)
  expect_equal(el$to, edges$dst)
  expect_equal(el$weight, edges$w)
  expect_error(graph_from_edgelist(data.frame(from = 1, to = 7), nodes = data.frame(id = 1:2)),
               "missing from")
})

test_that("adjacency conversions agree, dense and sparse", {
  A <- matrix(c(0, 1, 0,
                1, 0, 2,
                0, 2, 3), 3, byrow = TRUE)
  g <- graph_from_adjacency(A)
  expect_false(g$directed)
  expect_equal(num_edges(g, TRUE), 3L)
  expect_equal(as.matrix(adjacency_from_graph(g)), A, ignore_attr = TRUE)
  gs <- graph_from_adjacency(Matrix::Matrix(A, sparse = TRUE))
  expect_equal(gs$edge_index, g$edge_index)
  D <- matrix(c(0, 1, 0, 0), 2)
  gd <- graph_from_adjacency(D)
  expect_true(gd$directed)
  expect_equal(gd$edge_index, rbind(2L, 1L))
  expect_equal(adjacency_from_graph(gd, sparse = FALSE), D, ignore_attr = TRUE)
})

test_that("graph_tensors keeps one based indices and label types", {
  skip_if_no_torch()
  g <- graph_data(x = diag(3), edge_index = rbind(c(1, 2), c(2, 3)), y = factor(c("a", "b", "a")),
                  train_mask = c(TRUE, FALSE, TRUE))
  tt <- graph_tensors(g)
  expect_equal(as.matrix(as.array(tt$edge_index)), g$edge_index, ignore_attr = TRUE)
  expect_true(tt$y$dtype == torch::torch_long())
  expect_equal(as.integer(as.array(tt$y)), c(1L, 2L, 1L))
  expect_true(tt$train_mask$dtype == torch::torch_bool())
  e <- graph_tensors(graph_data(x = diag(2)))
  expect_equal(e$edge_index$size(2), 0)
  expect_output(print(tt), "graph_tensors")
})

test_that("graph_from_adjacency reads every stored triangle and pattern matrices", {
  A <- matrix(0, 3, 3)
  A[1, 2] <- A[2, 1] <- A[2, 3] <- A[3, 2] <- 1
  S <- Matrix::forceSymmetric(Matrix::Matrix(A, sparse = TRUE), uplo = "L")
  expect_equal(num_edges(graph_from_adjacency(S)), num_edges(graph_from_adjacency(A)))
  expect_equal(num_edges(graph_from_adjacency(Matrix::Matrix(A, sparse = TRUE), directed = TRUE)), 4L)
  P <- Matrix::Matrix(A, sparse = TRUE) != 0
  expect_equal(num_edges(graph_from_adjacency(P)), num_edges(graph_from_adjacency(A)))
})
