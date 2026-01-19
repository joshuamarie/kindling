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

## Value

A `vctrs` vector with class "activation_spec" containing validated
activation specifications.
