# Structural and positional encodings

Node level encodings that give attention models information about graph
structure.

## Usage

``` r
laplacian_pe(g, k = 8L, use_rspectra = TRUE)

random_walk_pe(g, k = 8L)

shortest_path_distances(g, max_dist = 10L, directed = FALSE)

random_sign_flip(pe, seed = NULL)

add_positional_encoding(g, type = c("random_walk", "laplacian"), k = 8L)
```

## Arguments

- g:

  A `graph_data` object.

- k:

  Number of encoding dimensions.

- use_rspectra:

  Use `RSpectra` sparse eigensolvers (if installed) for graphs larger
  than 500 nodes.

- max_dist:

  Distances above this are clipped to `max_dist + 1`.

- directed:

  Follow edge direction.

- pe:

  A positional encoding matrix.

- seed:

  Optional integer seed.

- type:

  `"laplacian"` or `"random_walk"`.

## Value

An N by `k` matrix (`laplacian_pe()`, `random_walk_pe()`) or an N by N
integer matrix.

## Details

- `laplacian_pe()`: the `k` eigenvectors of the symmetric normalized
  Laplacian \\L = I - D^{-1/2} A D^{-1/2}\\ with the smallest nonzero
  eigenvalues (Dwivedi et al. 2020). Direction is ignored (the adjacency
  is symmetrized). Eigenvectors are defined only up to sign, and up to a
  rotation within repeated eigenvalues, so the encoding is **not**
  permutation equivariant as a function: a permuted graph can give
  different signs. Train with random sign flips (`random_sign_flip()`)
  to make a model robust to this. Graphs with fewer than `k + 1` nodes
  are padded with zero columns.

- `random_walk_pe()`: the return probabilities \\\[(D^{-1}A)^t\]\_{ii}\\
  for \\t = 1, \dots, k\\ (Dwivedi et al. 2022). This encoding is
  permutation equivariant and has no sign ambiguity. It forms dense
  matrix powers, \\O(k n^3)\\ time.

- `shortest_path_distances()`: the all pairs hop distance matrix, with
  unreachable pairs set to `max_dist + 1`; used as attention bias by
  [`graph_transformer_model()`](https://adamrsophie.github.io/graphlearnR/reference/graph_transformer_model.md).
  Equivariant: permuting nodes permutes rows and columns.

`add_positional_encoding()` appends the encoding to the node features of
`g` (or of every graph in a list).

## Examples

``` r
g <- dataset_karate()
dim(laplacian_pe(g, 4))
#> [1] 34  4
dim(random_walk_pe(g, 3))
#> [1] 34  3
shortest_path_distances(g)[1:3, 1:3]
#>      [,1] [,2] [,3]
#> [1,]    0    1    1
#> [2,]    1    0    1
#> [3,]    1    1    0
```
