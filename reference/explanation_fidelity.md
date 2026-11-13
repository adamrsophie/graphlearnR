# Fidelity and stability of an explanation

`explanation_fidelity()` removes the `k` edges with the largest absolute
attribution (\\fid^{+}\\, the drop in the explained output; large is
good) and, separately, keeps only those `k` edges (\\fid^{-}\\, the drop
when everything else is removed; small is good), following Yuan et al.
(2022). `explanation_stability()` reruns an explainer on graphs with
small Gaussian feature noise and reports the mean Spearman correlation
of edge attributions with the original.

## Usage

``` r
explanation_fidelity(model, g, explanation, k = 5L)

explanation_stability(
  model,
  g,
  explainer,
  noise = 0.01,
  reps = 5L,
  seed = NULL
)
```

## Arguments

- model:

  A trained model or `graph_fit`.

- g:

  A `graph_data` object.

- explanation:

  A `graph_explanation` with an `edges` table.

- k:

  Number of top edges.

- explainer:

  Function `(model, g)` returning a `graph_explanation`.

- noise:

  Standard deviation of feature noise.

- reps:

  Number of noisy repetitions.

- seed:

  Optional integer seed.

## Value

A named numeric vector.
