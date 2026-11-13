# Split graphs for graph level prediction

Returns indices of training, validation and test graphs.

## Usage

``` r
split_graphs(
  n,
  train = 0.8,
  val = 0.1,
  test = 0.1,
  method = c("random", "group", "scaffold"),
  labels = NULL,
  groups = NULL,
  seed = NULL
)
```

## Arguments

- n:

  Number of graphs, or the dataset (its length is used).

- train, val, test:

  Proportions.

- method:

  Split method.

- labels:

  Optional labels for stratified random splits.

- groups:

  Group or scaffold identifiers, one per graph.

- seed:

  Optional integer seed.

## Value

A list with integer vectors `train`, `val`, `test`.

## Details

- `"random"`: random split, stratified by `labels` when given.

- `"group"`: group disjoint split; all graphs of a group go to the same
  set (for example several conformers of one molecule, or graphs from
  the same patient). Groups are shuffled, then assigned in order until
  each set reaches its target size.

- `"scaffold"`: deterministic scaffold split as in MoleculeNet (Wu et
  al. 2018): `groups` holds a scaffold identifier per molecule (for
  example a Bemis-Murcko scaffold computed with RDKit); scaffold sets
  are sorted from largest to smallest and assigned to training first,
  then validation, then test, so test molecules have scaffolds not seen
  in training. graphlearnR does not compute scaffolds itself.

## Examples

``` r
split_graphs(10, seed = 1)
#> $train
#> [1]  9  4  7  1  2  5  3 10
#> 
#> $val
#> [1] 6
#> 
#> $test
#> [1] 8
#> 
split_graphs(6, method = "scaffold", groups = c("a", "a", "a", "b", "b", "c"))
#> $train
#> [1] 1 2 3 6
#> 
#> $val
#> [1] 4 5
#> 
#> $test
#> integer(0)
#> 
```
