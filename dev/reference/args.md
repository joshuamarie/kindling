# Activation Function Arguments Helper

**\[superseded\]**

This is superseded in v0.3.0 in favour of indexed syntax, e.g.
`<act_fn[param = 0]>` type. Type-safe helper to specify parameters for
activation functions. All parameters must be named and match the formal
arguments of the corresponding `{torch}` activation function.

## Usage

``` r
args(...)
```

## Arguments

- ...:

  Named arguments for the activation function.

## Value

A list with class "activation_args" containing the parameters.
