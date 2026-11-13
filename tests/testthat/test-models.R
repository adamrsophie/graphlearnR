test_that("gnn_model builds every architecture with the right output shape", {
  skip_if_no_torch()
  g <- random_graph(n = 8, m = 16, f = 3, seed = 1)
  g$edge_attr <- matrix(withr::with_seed(1, rnorm(32)), 16, 2)
  g$edge_type <- withr::with_seed(1, sample(1:2, 16, TRUE))
  tt <- graph_tensors(g)
  for (a in c("sage", "gcn", "gat", "gatv2", "gin", "graph_conv", "transformer", "mpnn", "rgcn")) {
    m <- gnn_model(a, 3, 8, 4, heads = 2, edge_dim = 2, num_relations = 2, norm = "layer",
                   residual = TRUE)
    m$eval()
    out <- graphlearnR:::model_forward(m, tt)
    expect_equal(dim(out), c(8, 4), info = a)
    emb <- graphlearnR:::model_forward(m, tt, return_embedding = TRUE)
    expect_equal(dim(emb), c(8, 8), info = a)
  }
  enc <- gnn_model("sage", 3, 8)
  expect_equal(dim(graphlearnR:::model_forward(enc, tt)), c(8, 8))
  expect_error(gnn_model("gat", 3, 10, heads = 4), "divisible")
  expect_error(gnn_model("mpnn", 3, 8), "edge_dim")
  expect_error(gnn_model("rgcn", 3, 8), "num_relations")
})

test_that("graph level models are permutation invariant", {
  skip_if_no_torch()
  torch::torch_manual_seed(3)
  g <- random_graph(n = 9, m = 18, f = 2, seed = 3)
  perm <- withr::with_seed(4, sample.int(9))
  gp <- permute_graph(g, perm)
  for (pool in c("mean", "sum", "max", "attention")) {
    for (a in c("gin", "gcn", "gat")) {
      m <- gnn_model(a, 2, 8, 3, heads = 2, pool = pool)
      m$eval()
      o1 <- as_mat(graphlearnR:::model_forward(m, graph_tensors(g)))
      o2 <- as_mat(graphlearnR:::model_forward(m, graph_tensors(gp)))
      expect_equal(o2, o1, tolerance = 1e-5, info = paste(a, pool))
    }
  }
})

test_that("node level models are permutation equivariant", {
  skip_if_no_torch()
  torch::torch_manual_seed(4)
  g <- random_graph(n = 9, m = 18, f = 2, seed = 5)
  perm <- withr::with_seed(6, sample.int(9))
  gp <- permute_graph(g, perm)
  m <- gnn_model("sage", 2, 8, 3, num_layers = 3)
  m$eval()
  o1 <- as_mat(graphlearnR:::model_forward(m, graph_tensors(g)))
  o2 <- as_mat(graphlearnR:::model_forward(m, graph_tensors(gp)))
  expect_equal(o2, o1[perm, ], tolerance = 1e-5)
})

test_that("a batched forward pass equals separate forward passes", {
  skip_if_no_torch()
  torch::torch_manual_seed(5)
  gs <- lapply(1:4, function(s) random_graph(n = 4 + s, m = 8, f = 2, seed = 10 + s))
  b <- graph_batch(gs)
  m <- gnn_model("gat", 2, 8, 2, heads = 2, pool = "mean")
  m$eval()
  ob <- as_mat(graphlearnR:::model_forward(m, graph_tensors(b)))
  os <- do.call(rbind, lapply(gs, function(g) as_mat(graphlearnR:::model_forward(m, graph_tensors(g)))))
  expect_equal(ob, os, tolerance = 1e-5)
})

test_that("gradients reach every parameter through batching and pooling", {
  skip_if_no_torch()
  b <- graph_batch(dataset_synthetic_graphs(5, seed = 1))
  m <- gnn_model("gin", 1, 8, 2, pool = "sum")
  tt <- graph_tensors(b)
  loss <- torch::nnf_cross_entropy(graphlearnR:::model_forward(m, tt), tt$graph_y)
  loss$backward()
  ok <- vapply(m$parameters, function(p) !is.null(p$grad) && as.numeric(p$grad$abs()$sum()$item()) > 0,
               logical(1))
  expect_true(all(ok))
})

test_that("decoders score pairs", {
  skip_if_no_torch()
  z <- torch::torch_randn(5, 4)
  pairs <- torch::torch_tensor(rbind(c(1, 2), c(3, 4)), dtype = torch::torch_long())
  zz <- as_mat(z)
  expect_equal(as.numeric(as.array(link_decoder("dot")(z, pairs))),
               c(sum(zz[1, ] * zz[3, ]), sum(zz[2, ] * zz[4, ])), tolerance = 1e-5)
  for (tp in c("distmult", "mlp", "hadamard")) {
    expect_equal(length(as.array(link_decoder(tp, dim = 4)(z, pairs))), 2)
  }
  expect_equal(dim(edge_decoder(4, 3)(z, pairs)), c(2, 3))
  expect_error(link_decoder("mlp"), "dim")
})

test_that("topk_pool keeps the top nodes per graph and their edges", {
  skip_if_no_torch()
  b <- graph_batch(list(random_graph(6, 12, 2, 1), random_graph(4, 8, 2, 2)))
  tt <- graph_tensors(b)
  pool <- topk_pool(2, ratio = 0.5)
  res <- pool(tt$x, tt$edge_index, tt$batch)
  expect_equal(dim(res$x), c(5, 2))
  expect_equal(as.integer(as.array(res$batch)), c(1L, 1L, 1L, 2L, 2L))
  kept <- res$perm
  ei <- b$edge_index
  expect_equal(ncol(as_mat(res$edge_index)), sum(ei[1, ] %in% kept & ei[2, ] %in% kept))
})
