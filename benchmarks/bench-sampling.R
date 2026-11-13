# Times sampling separately from model execution: building the mini batches
# of one epoch (neighbor, Cluster-GCN, GraphSAINT and LADIES loaders) and, on the
# same prebuilt batches, the forward and backward passes of a GraphSAGE
# model. CPU only.
# Usage: Rscript benchmarks/bench-sampling.R <output_dir>
suppressPackageStartupMessages({
  library(graphlearnR)
  library(torch)
})
out_dir <- commandArgs(TRUE)[1]
if (is.na(out_dir)) out_dir <- "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

rows <- list()
for (n_per in c(500, 2500)) {
  g <- split_nodes(dataset_sbm(rep(n_per, 4), p_in = 20 / n_per, p_out = 2 / n_per, seed = 1), seed = 1)
  loaders <- list(
    neighbor = node_loader(g, batch_size = 256, fanouts = c(10, 10), seed = 1),
    cluster = cluster_loader(g, num_parts = 16, clusters_per_batch = 2, seed = 1),
    saint = saint_loader(g, batch_size = 200, walk_length = 2, num_steps = 10, seed = 1),
    ladies = ladies_loader(g, batch_size = 256, layer_sizes = c(512, 512), seed = 1)
  )
  for (nm in names(loaders)) {
    t_sample <- system.time(batches <- loader_batches(loaders[[nm]], epoch = 1))[["elapsed"]]
    torch_manual_seed(1)
    m <- gnn_model("sage", ncol(g$x), 64, 4)
    opt <- optim_adam(m$parameters, lr = 0.01)
    t_model <- system.time(for (b in batches) {
      tt <- graph_tensors(b)
      opt$zero_grad()
      out <- m(tt$x, tt$edge_index)
      loss <- nnf_cross_entropy(out, tt$y)
      loss$backward()
      opt$step()
    })[["elapsed"]]
    rows[[length(rows) + 1L]] <- data.frame(
      benchmark = "sampling_vs_model", loader = nm, nodes = num_nodes(g), edges = num_edges(g),
      batches = length(batches), sampling_seconds = t_sample, model_seconds = t_model)
  }
}
res <- do.call(rbind, rows)
print(res, row.names = FALSE)
cat("\n"); print(sessionInfo()$running); cat(parallel::detectCores(), "cores\n")
utils::write.csv(res, file.path(out_dir, "sampling.csv"), row.names = FALSE)
