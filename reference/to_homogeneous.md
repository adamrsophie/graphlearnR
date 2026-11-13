# Convert a heterogeneous graph to a homogeneous one

Concatenates node types in the order of `hg$node_types` and relations in
the order of `names(hg$edges)`. `node_type` and `edge_type` become
factors with these levels; `metadata$type_ptr` gives node offsets, so
node `i` of type `t` becomes global node `type_ptr[t] + i`. Feature
matrices of different sizes are padded with zeros to the largest size,
and `metadata$feature_dims` records the original sizes. Labels and masks
come from `target_type` only (other nodes get `NA` labels and `FALSE`
masks).

## Usage

``` r
to_homogeneous(hg, target_type = NULL)
```

## Arguments

- hg:

  A `hetero_graph_data` object.

- target_type:

  Node type whose labels and masks are kept.

## Value

A directed `graph_data` object.

## Examples

``` r
g <- to_homogeneous(dataset_synthetic_hetero(seed = 1), target_type = "drug")
g
#> <graph_data> directed graph: 140 nodes, 534 edges
#>   node features [140 x 6]
#>   node labels: factor with 3 levels
#>   masks: train=20, val=20, test=20
```
