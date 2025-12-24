# Depth-Aware Grid Generation for Neural Networks

`grid_depth()` extends standard grid generation to support multi-layer
neural network architectures. It creates heterogeneous layer
configurations by generating list columns for `hidden_neurons` and
`activations`.

## Usage

``` r
grid_depth(
  x,
  ...,
  n_hlayer = 2L,
  size = 5L,
  type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
  original = TRUE,
  levels = 3L,
  variogram_range = 0.5,
  iter = 1000L
)

# S3 method for class 'parameters'
grid_depth(
  x,
  ...,
  n_hlayer = 2L,
  size = 5L,
  type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
  original = TRUE,
  levels = 3L,
  variogram_range = 0.5,
  iter = 1000L
)

# S3 method for class 'list'
grid_depth(
  x,
  ...,
  n_hlayer = 2L,
  size = 5L,
  type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
  original = TRUE,
  levels = 3L,
  variogram_range = 0.5,
  iter = 1000L
)

# S3 method for class 'workflow'
grid_depth(
  x,
  ...,
  n_hlayer = 2L,
  size = 5L,
  type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
  original = TRUE,
  levels = 3L,
  variogram_range = 0.5,
  iter = 1000L
)

# S3 method for class 'model_spec'
grid_depth(
  x,
  ...,
  n_hlayer = 2L,
  size = 5L,
  type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
  original = TRUE,
  levels = 3L,
  variogram_range = 0.5,
  iter = 1000L
)

# S3 method for class 'param'
grid_depth(
  x,
  ...,
  n_hlayer = 2L,
  size = 5L,
  type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
  original = TRUE,
  levels = 3L,
  variogram_range = 0.5,
  iter = 1000L
)

# Default S3 method
grid_depth(
  x,
  ...,
  n_hlayer = 2L,
  size = 5L,
  type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
  original = TRUE,
  levels = 3L,
  variogram_range = 0.5,
  iter = 1000L
)
```

## Arguments

- x:

  A `parameters` object, list, workflow, or model spec. Can also be a
  single `param` object if `...` contains additional param objects.

- ...:

  One or more `param` objects (e.g.,
  [`hidden_neurons()`](https://packages.joshuamarie.com/kindling/reference/dials-kindling.md),
  [`epochs()`](https://dials.tidymodels.org/reference/dropout.html)). If
  `x` is a `parameters` object, `...` is ignored. None of the objects
  can have
  [`unknown()`](https://dials.tidymodels.org/reference/unknown.html)
  values.

- n_hlayer:

  Integer vector specifying number of hidden layers to generate (e.g.,
  `2:4` for 2, 3, or 4 layers). Default is 2.

- size:

  Integer. Number of parameter combinations to generate.

- type:

  Character. Type of grid: "regular", "random", "latin_hypercube",
  "max_entropy", or "audze_eglais".

- original:

  Logical. Should original parameter ranges be used?

- levels:

  Integer. Levels per parameter for regular grids.

- variogram_range:

  Numeric. Range for audze_eglais design.

- iter:

  Integer. Iterations for max_entropy optimization.

## Value

A tibble with list columns for `hidden_neurons` and `activations`, where
each element is a vector of length `n_hlayer`.

## Details

This function is specifically for `{kindling}` models. The `n_hlayer`
parameter determines network depth and creates list columns for
`hidden_neurons` and `activations`, where each element is a vector of
length matching the sampled depth.

## Examples

``` r
if (FALSE) { # \dontrun{
library(dials)

# Method 1: Using parameters()
params = parameters(
    hidden_neurons(c(32L, 128L)),
    activations(c("relu", "elu", "selu")),
    epochs(c(50L, 200L))
)
grid = grid_depth(params, n_hlayer = 2:3, type = "regular", levels = 3)

# Method 2: Direct param objects
grid = grid_depth(
    hidden_neurons(c(32L, 128L)),
    activations(c("relu", "elu")),
    epochs(c(50L, 200L)),
    n_hlayer = 2:3,
    type = "random",
    size = 20
)

# Method 3: From workflow
wf = workflow() |>
    add_model(mlp_kindling(hidden_neurons = tune(), activations = tune())) |>
    add_formula(y ~ .)
grid = grid_depth(wf, n_hlayer = 2:4, type = "latin_hypercube", size = 15)
} # }
```
