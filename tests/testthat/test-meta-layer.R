test_that("meta_layer updates edges, nodes and graphs as written", {
  skip_if_no_torch()
  g <- random_graph(n = 6, m = 10, f = 2, seed = 3)
  tt <- graph_tensors(g, dtype = torch::torch_double())
  ea <- torch::torch_randn(10, 1, dtype = torch::torch_double())
  u <- torch::torch_randn(1, 2, dtype = torch::torch_double())
  block <- meta_layer(
    edge_model = function(x_src, x_dst, edge_attr, u, ...) {
      (x_src - x_dst)$sum(dim = 2, keepdim = TRUE) + edge_attr + u[, 1:1]
    },
    node_model = function(x, edge_index, edge_attr, ...) {
      x + scatter_sum(edge_attr, edge_index[2, ], x$size(1))
    },
    global_model = function(x, batch, ...) global_add_pool(x, batch)
  )
  out <- block(tt$x, tt$edge_index, edge_attr = ea, u = u)
  ei <- g$edge_index
  e_ref <- rowSums(g$x[ei[1, ], , drop = FALSE] - g$x[ei[2, ], , drop = FALSE]) +
    as.numeric(as.array(ea)) + as.numeric(as.array(u))[1]
  expect_equal(as.numeric(as.array(out$edge_attr)), e_ref, tolerance = 1e-10)
  agg <- as.numeric(tapply(c(e_ref, rep(0, 6)), c(ei[2, ], 1:6), sum))
  x_ref <- g$x + agg
  expect_equal(as_mat(out$x), x_ref, tolerance = 1e-10)
  expect_equal(as.numeric(as.array(out$u)), colSums(x_ref), tolerance = 1e-10)
})

test_that("meta_layer registers module parameters and is permutation equivariant", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  block <- meta_layer(
    edge_model = torch::nn_module(
      initialize = function() self$lin <- torch::nn_linear(4, 3),
      forward = function(x_src, x_dst, ...) self$lin(torch::torch_cat(list(x_src, x_dst), dim = 2))
    )(),
    node_model = function(x, edge_index, edge_attr, ...) {
      torch::torch_cat(list(x, scatter_mean(edge_attr, edge_index[2, ], x$size(1))), dim = 2)
    }
  )
  expect_length(block$parameters, 2L)
  g <- random_graph(n = 7, m = 12, f = 2, seed = 4)
  perm <- c(3, 1, 7, 2, 5, 6, 4)
  gp <- permute_graph(g, perm)
  a <- as_mat(block(graph_tensors(g)$x, graph_tensors(g)$edge_index)$x)
  b <- as_mat(block(graph_tensors(gp)$x, graph_tensors(gp)$edge_index)$x)
  expect_equal(b, a[perm, ], tolerance = 1e-5)
})
