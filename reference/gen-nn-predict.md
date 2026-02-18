# Predict method for nn_fit objects

Predict method for nn_fit objects

## Usage

``` r
# S3 method for class 'nn_fit'
predict(object, newdata = NULL, new_data = NULL, type = "response", ...)

# S3 method for class 'nn_fit_tab'
predict(object, newdata = NULL, new_data = NULL, type = "response", ...)
```

## Arguments

- object:

  An object of class `"nn_fit"` or `"nn_fit_tab"`.

- newdata:

  Data frame or matrix. New data for predictions.

- new_data:

  Alternative to `newdata` (hardhat-style).

- type:

  Character. `"response"` (default) or `"prob"` (classification only).

- ...:

  Currently unused.

## Value

Numeric vector/matrix (regression) or factor / probability matrix
(classification).
