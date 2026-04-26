# Plot training loss history for a fitted neural network

Creates a line plot of training (and optionally validation) loss over
epochs for a model of class `nn_fit`.

## Usage

``` r
# S3 method for class 'nn_fit'
autoplot(object, ...)

# S3 method for class 'nn_fit'
plot(x, ...)
```

## Arguments

- object:

  A fitted model of class `nn_fit`, as returned by
  [`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md),
  [`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md),
  or
  [`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md).

- ...:

  Additional arguments (currently unused).

- x:

  A fitted model of class `nn_fit`, as returned by
  [`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md),
  [`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md),
  or
  [`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md).

## Value

A
[`ggplot2::ggplot()`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object showing loss vs epoch.

## Examples

``` r
# \donttest{
  if (torch::torch_is_installed()) {
    m = train_nn(
      as.matrix(iris[, 2:4]), iris$Sepal.Length,
      epochs = 5
    )
    ggplot2::autoplot(m)
  }

# }
```
