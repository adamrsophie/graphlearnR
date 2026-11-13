accelerator <- function() {
  if (torch::cuda_is_available()) return("cuda")
  if (torch::backends_mps_is_available()) return("mps")
  NULL
}

test_that("CPU and accelerator forward passes agree within float tolerance", {
  skip_if_no_torch()
  dev <- accelerator()
  skip_if(is.null(dev), "No CUDA or MPS device available")
  # The virtual GPU of GitHub's macOS runners gave outputs unrelated to the CPU
  # ones, while an Apple M2 agreed to 1e-7; check MPS on real hardware only.
  if (dev == "mps") skip_on_ci()
  torch::torch_manual_seed(1)
  g <- random_graph(30, 90, 4, seed = 1)
  for (a in c("sage", "gcn", "gat")) {
    m <- gnn_model(a, 4, 8, 3, heads = 2)
    m$eval()
    cpu <- as_mat(graphlearnR:::model_forward(m, graph_tensors(g, "cpu")))
    m$to(device = dev)
    acc <- as_mat(graphlearnR:::model_forward(m, graph_tensors(g, dev)))
    expect_equal(acc, cpu, tolerance = 1e-4, info = a)
  }
})

test_that("devices resolve and unavailable devices fail clearly", {
  skip_if_no_torch()
  expect_equal(graphlearnR:::resolve_device(NULL)$type, "cpu")
  expect_true(graphlearnR:::resolve_device("auto")$type %in% c("cpu", "cuda", "mps"))
  if (!torch::cuda_is_available()) expect_error(graphlearnR:::resolve_device("cuda"), "not available")
})
