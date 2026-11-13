# Synthetic heterogeneous drug and protein graph

A small simulated graph with `drug` and `protein` nodes for examples and
tests. Proteins belong to latent families; drugs belong to latent
classes and preferentially target proteins of the family linked to their
class. Drug labels are the latent class, so they can be predicted from
the targeted proteins. Protein-protein interactions are denser within
families. The data are simulated and carry no biological meaning.

## Usage

``` r
dataset_synthetic_hetero(
  num_drugs = 60L,
  num_proteins = 80L,
  num_classes = 3L,
  seed = NULL
)
```

## Arguments

- num_drugs, num_proteins:

  Node counts.

- num_classes:

  Latent classes (drug labels and protein families).

- seed:

  Optional integer seed.

## Value

A `hetero_graph_data` with relations `drug__targets__protein` and
`protein__interacts__protein`, drug labels and drug masks.
