# Ordinal Suffixes Generator

This function is originally from `numform::f_ordinal()`.

## Usage

``` r
ordinal_gen(x)
```

## Arguments

- x:

  Vector of numbers. Could be a string equivalent

## Value

Returns a string vector with ordinal suffixes.

## This is how you use it

    kindling:::ordinal_gen(1:10)

## References

Rinker, T. W. (2021). numform: A publication style number and plot
formatter version 0.7.0. <https://github.com/trinker/numform>

## Examples

``` r
kindling:::ordinal_gen(1:10)
#>  [1] "1st"  "2nd"  "3rd"  "4th"  "5th"  "6th"  "7th"  "8th"  "9th"  "10th"

# Note: this function is not exported into the public namespace.
# Refer to `numform::f_ordinal()` instead.
```
