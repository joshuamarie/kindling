# Activation Function Arguments Helper

Type-safe helper to specify parameters for activation functions. All
parameters must be named and match the formal arguments of the
corresponding `torch` activation function.

## Usage

``` r
args(...)
```

## Arguments

- ...:

  Named arguments for the activation function.

## Value

A list with class "activation_args" containing the parameters.
