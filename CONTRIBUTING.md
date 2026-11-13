# Contributing to graphlearnR

Thank you for considering a contribution to `graphlearnR`. The package brings
graph machine learning and graph neural networks to R on top of `torch`. A
change is judged on whether it is numerically correct, whether it keeps the
graph semantics the package documents, and whether it fits R conventions.

## Before You Start

- Use GitHub issues for bug reports, feature requests, and documentation gaps.
- For a new layer, task, or dataset, open an issue first so the design can be
  discussed before implementation.
- Keep pull requests focused. A new layer, a bug fix, and a refactor should
  usually be separate pull requests.

## Development Setup

```r
install.packages(c("remotes", "devtools", "roxygen2", "testthat"))
remotes::install_deps(dependencies = TRUE)
torch::install_torch()
```

Tests that need LibTorch skip when it is not installed. Tests that need a GPU
skip when no CUDA or MPS device is available.

## Package Rules

- Node indices are 1-based in every public interface: `edge_index`, seeds,
  pairs, masks, and sampled node ids. Convert to 0-based only through
  `to_zero_based()` at an explicit boundary.
- An undirected graph is given with one entry per pair and stored in both
  directions. Code that counts, splits, or samples edges must respect
  `edge_pair`.
- Splits must not leak. A held out link must not appear in the message
  passing graph of any training step, nor in a reverse relation of a
  heterogeneous graph. Temporal code must only use events strictly before the
  query time.
- New layers need a permutation equivariance test, a comparison with a dense
  or reference formula, and a finite difference gradient check in double
  precision.
- Reproducibility: functions that draw random numbers take a `seed` argument
  and must not change the caller's R random number state.
- Do not add performance claims without a benchmark script under
  `benchmarks/` and recorded hardware details.
- Do not bundle third party datasets unless their license allows
  redistribution. Download them with checksum verification instead.

## Branches and Pull Requests

- Use one branch per logical change, named after the change
  (`add-cheb-conv`, `fix-negative-sampling-bipartite`).
- Write commit subjects in the imperative ("Add Chebyshev spectral
  convolution").
- Keep the implementation, its tests, the roxygen source, the regenerated
  `man/` pages, and a NEWS entry in the same pull request.
- Continuous integration runs `R CMD check` on Linux, macOS, and Windows. A
  pull request is ready to merge when the checks pass.

## Code of Conduct

Please note that this project is released with a
[Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree
to abide by its terms.
