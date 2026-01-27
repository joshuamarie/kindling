# Basemodels-tidymodels wrappers

Basemodels-tidymodels wrappers

## Usage

``` r
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

- `ffnn_wrapper()` returns an object of class `"ffnn_fit"` containing
  the trained feedforward neural network model and metadata. See
  [`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
  for details.

- `rnn_wrapper()` returns an object of class `"rnn_fit"` containing the
  trained recurrent neural network model and metadata. See
  [`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
  for details.

## Details

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

## FFNN (MLP) Wrapper for `{tidymodels}` interface

This is a function to interface into `{tidymodels}` (do not use this,
use
[`kindling::ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
instead).

## RNN Wrapper for `{tidymodels}` interface

This is a function to interface into `{tidymodels}` (do not use this,
use
[`kindling::rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
instead).
