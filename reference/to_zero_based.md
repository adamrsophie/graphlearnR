# Convert one based indices to zero based

The single place where graphlearnR converts its one based node indices
to the zero based convention used by Python graph libraries and some
file formats. `from_zero_based()` is the inverse.

## Usage

``` r
to_zero_based(index)

from_zero_based(index)
```

## Arguments

- index:

  Integer vector or matrix of one based indices.

## Value

Integer vector or matrix of the same shape.

## Examples

``` r
to_zero_based(rbind(c(1, 2), c(2, 3)))
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    1    2
```
