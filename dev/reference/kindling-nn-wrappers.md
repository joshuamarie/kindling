# kindling-tidymodels wrapper

kindling-tidymodels wrapper

Basemodels-tidymodels wrappers

## Usage

``` r
train_nn_wrapper(formula, data, ...)

ffnn_wrapper(formula, data, ...)

rnn_wrapper(formula, data, ...)
```

## Arguments

- formula:

  A formula specifying the model (e.g., `y ~ x1 + x2`)

- data:

  A data frame containing the training data

- ...:

  Additional arguments passed to the underlying training function

## Value

`train_nn_wrapper()` returns an `"nn_fit_tab"` object. See
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
for details.

- `ffnn_wrapper()` returns an object of class `"ffnn_fit"` containing
  the trained feedforward neural network model and metadata. See
  [`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
  for details.

- `rnn_wrapper()` returns an object of class `"rnn_fit"` containing the
  trained recurrent neural network model and metadata. See
  [`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
  for details.

## Details

This wrapper function is designed to interface with the `{tidymodels}`
ecosystem, particularly for use with
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html)
and workflows. It handles the conversion of tuning parameters
(especially list-column parameters from
[`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md))
into the format expected by
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md).

These wrapper functions are designed to interface with the
`{tidymodels}` ecosystem, particularly for use with
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html)
and workflows. They handle the conversion of tuning parameters
(especially list-column parameters from
[`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md))
into the format expected by
[`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
and
[`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md).

## MLP Wrapper for `{tidymodels}` interface

Internal wrapper — use
[`mlp_kindling()`](https://kindling.joshuamarie.com/dev/reference/mlp_kindling.md) +
[`fit()`](https://generics.r-lib.org/reference/fit.html) instead.

## FFNN (MLP) Wrapper for `{tidymodels}` interface

This is a function to interface into `{tidymodels}` (do not use this,
use
[`kindling::ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
instead).

## RNN Wrapper for `{tidymodels}` interface

Internal wrapper — use
[`rnn_kindling()`](https://kindling.joshuamarie.com/dev/reference/rnn_kindling.md) +
[`fit()`](https://generics.r-lib.org/reference/fit.html) instead.
