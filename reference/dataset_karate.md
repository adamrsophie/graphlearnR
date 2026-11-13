# Zachary's karate club

The social network of 34 members of a university karate club observed by
Zachary (1977), with the two factions that formed after the club split
(17 members followed the instructor, "Mr. Hi"; 17 the administrator,
"Officer"). Edges come from `igraph::make_graph("Zachary")`; faction
labels follow the `club` attribute distributed with NetworkX. Node
features are one-hot node indicators (34 columns), the featureless
setting of Kipf and Welling (2017).

## Usage

``` r
dataset_karate(features = c("identity", "degree", "none"))
```

## Arguments

- features:

  `"identity"` for one-hot indicators, `"degree"` for a single degree
  column, or `"none"`.

## Value

An undirected `graph_data` with 34 nodes, 78 edges and factor labels
`y`.

## References

Zachary, W. W. (1977). An information flow model for conflict and
fission in small groups. Journal of Anthropological Research, 33(4), 452
to 473.

## Examples

``` r
g <- dataset_karate()
table(g$y)
#> 
#>  Mr. Hi Officer 
#>      17      17 
```
