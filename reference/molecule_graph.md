# Molecular graphs from atoms and bonds

`molecule_graph()` builds an undirected
[`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md)
object with one node per atom and one edge pair per bond. Node features
are a one hot encoding of the element over `elements` (an extra "other"
column catches anything else); edge features are a one hot encoding of
the bond order over `bond_orders`, repeated for both directions.

## Usage

``` r
molecule_graph(
  atoms,
  bonds,
  elements = c("C", "N", "O", "S", "F", "Cl", "Br", "P", "I"),
  bond_orders = c(1, 2, 3, 1.5),
  graph_y = NULL
)

dataset_synthetic_molecules(n = 100L, seed = NULL)
```

## Arguments

- atoms:

  Character vector of element symbols.

- bonds:

  Integer matrix or data frame with columns `from`, `to` (1 based atom
  positions) and optionally `order` (bond order, default 1).

- elements:

  Elements given their own feature column.

- bond_orders:

  Bond orders given their own feature column (1.5 for aromatic).

- graph_y:

  Optional graph label.

- n:

  Number of molecules.

- seed:

  Optional seed; the caller's random stream is left untouched.

## Value

`molecule_graph()`: a `graph_data` object.
`dataset_synthetic_molecules()`: a list of them with a `"scaffold"`
attribute.

## Details

graphlearnR does not parse SMILES or compute Bemis-Murcko scaffolds; use
a chemistry toolkit such as RDKit for that and pass atoms, bonds and
scaffold identifiers here and to
[`split_graphs()`](https://adamrsophie.github.io/graphlearnR/reference/split_graphs.md).

`dataset_synthetic_molecules()` generates small random molecules: a
carbon backbone, optionally closed into a ring of 5 or 6 atoms, with
random heteroatoms (N, O) and side chains. The label `graph_y` is
`"ring"` or `"chain"`, and the list's `"scaffold"` attribute holds a
synthetic scaffold identifier per molecule (ring size and heteroatom
count in the ring) so that scaffold splits can be demonstrated. The data
are synthetic and carry no chemical meaning.

## Examples

``` r
# ethanol, heavy atoms only
g <- molecule_graph(c("C", "C", "O"), rbind(c(1, 2), c(2, 3)))
g
#> <graph_data> undirected graph: 3 nodes, 4 edges (2 undirected pairs)
#>   node features [3 x 10]
#>   edge features [4 x 4]
mols <- dataset_synthetic_molecules(20, seed = 1)
table(vapply(mols, function(m) as.character(m$graph_y), ""))
#> 
#> chain  ring 
#>    10    10 
```
