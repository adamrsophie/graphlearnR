# Times full-batch and neighbor-sampled training epochs and records accuracy
# on a stochastic block model.
# Usage: Rscript benchmarks/bench-training.R <output_dir>
suppressPackageStartupMessages({
  library(graphlearnR)
  library(torch)
})
out_dir <- commandArgs(TRUE)[1]
if (is.na(out_dir)) out_dir <- "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

rows <- list()
for (n_per in c(500, 2000)) {
  g <- dataset_sbm(rep(n_per, 4), p_in = 20 / n_per, p_out = 2 / n_per, seed = 1)
  g <- split_nodes(g, seed = 1)
  for (mode in c("full_batch", "sampled")) {
    torch_manual_seed(1)
    m <- gnn_model("sage", ncol(g$x), 64, 4, num_layers = 2)
    args <- list(m, g, "node_classification", epochs = 20, lr = 0.01, seed = 1)
    if (mode == "sampled") args <- c(args, list(fanouts = c(10, 10), batch_size = 256))
    t <- system.time(fit <- do.call(train_graph_model, args))[["elapsed"]]
    ev <- evaluate_graph_model(fit, g)
    rows[[length(rows) + 1L]] <- data.frame(
      benchmark = "train_epochs", mode = mode, nodes = num_nodes(g),
      edges = num_edges(g), epochs = 20, seconds = t, seconds_per_epoch = t / 20,
      test_accuracy = ev$value[ev$metric == "accuracy"])
  }
}
res <- do.call(rbind, rows)
print(res, row.names = FALSE)
utils::write.csv(res, file.path(out_dir, "training.csv"), row.names = FALSE)
