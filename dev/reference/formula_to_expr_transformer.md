# Convert Formula to Expression Transformer

Convert Formula to Expression Transformer

## Usage

``` r
formula_to_expr_transformer(formula_or_fn)
```

## Arguments

- formula_or_fn:

  A formula like `~ .[[1]]` or a function that transforms expressions

## Value

A function that takes an expression and returns a transformed
expression, or NULL
