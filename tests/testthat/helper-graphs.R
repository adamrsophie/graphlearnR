skip_if_no_torch <- function() {
  skip_if_not(torch::torch_is_installed(), "LibTorch is not installed")
}

random_graph <- function(n = 8, m = 16, f = 3, seed = 1, directed = TRUE) {
  withr::with_seed(seed, {
    ei <- rbind(sample.int(n, m, TRUE), sample.int(n, m, TRUE))
    x <- matrix(rnorm(n * f), n, f)
    w <- runif(m, 0.5, 1.5)
  })
  graph_data(x = x, edge_index = ei, edge_weight = w, num_nodes = n,
             directed = directed)
}

# Relabel nodes: new node i is old node perm[i].
permute_graph <- function(g, perm) {
  inv <- integer(length(perm))
  inv[perm] <- seq_along(perm)
  out <- g
  out$x <- g$x[perm, , drop = FALSE]
  if (!is.null(g$y)) out$y <- g$y[perm]
  out$edge_index <- matrix(inv[g$edge_index], nrow = 2L)
  for (m in c("train_mask", "val_mask", "test_mask")) {
    if (!is.null(g[[m]])) out[[m]] <- g[[m]][perm]
  }
  out
}

as_mat <- function(t) as.matrix(as.array(t$detach()$cpu()))
