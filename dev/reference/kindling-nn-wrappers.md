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
