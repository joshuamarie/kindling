# Activation Functions Specification Helper

This function is a DSL function, kind of like
[`ggplot2::aes()`](https://ggplot2.tidyverse.org/reference/aes.html),
that helps to specify activation functions for neural network layers. It
validates that activation functions exist in `torch` and that any
parameters match the function's formal arguments.

## Usage

``` r
act_funs(...)
```

## Arguments

- ...:

  Activation function specifications. Can be:

  - Bare symbols: `relu`, `tanh`

  - Character strings (simple): `"relu"`, `"tanh"`

  - Character strings (with params): `"softshrink(lambda = 0.1)"`,
    `"rrelu(lower = 1/5, upper = 1/4)"`

  - Named with parameters: `softmax = args(dim = 2L)`

  - Indexed syntax (named): `softshrink[lambd = 0.2]`,
    `rrelu[lower = 1/5, upper = 1/4]`

  - Indexed syntax (unnamed): `softshrink[0.5]`, `elu[0.5]`

## Value

A `vctrs` vector with class "activation_spec" containing validated
activation specifications.

## Examples

``` r
act_funs(relu, sigmoid)
#> <activation_spec[2]>
#>                 
#>    relu sigmoid 
act_funs(relu, softshrink[lambd = 0.5], elu)
#> <activation_spec[3]>
#>                
#> relu  0.5  elu 
act_funs(softmax = args(dim = 2L))
#> Warning: `args()` was deprecated in kindling 0.3.0.
#> ℹ Use indexed syntax for parametric activation functions, e.g. `<softplus[beta
#>   = 0.5]>`.
#> <activation_spec[1]>
#> softmax 
#>       2 
```
