# Formula to Function with Named Arguments

Formula to Function with Named Arguments

## Usage

``` r
formula_to_function(
  formula_or_fn,
  default_fn = NULL,
  arg_names = NULL,
  alias_map = NULL
)
```

## Arguments

- formula_or_fn:

  A formula or function

- default_fn:

  Default function if `formula_or_fn` is `NULL`

- arg_names:

  Character vector of formal argument names

- alias_map:

  Named list mapping arg_names to formula aliases (e.g., list(in_dim =
  ".in"))

## Value

A function
