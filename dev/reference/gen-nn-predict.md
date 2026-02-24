# Predict from a trained neural network

Generate predictions from an `"nn_fit"` object produced by
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md).

Three S3 methods are registered:

- `predict.nn_fit()` — base method for `matrix`-trained models.

- `predict.nn_fit_tab()` — extends the base method for tabular fits;
  runs new data through
  [`hardhat::forge()`](https://hardhat.tidymodels.org/reference/forge.html)
  before predicting.

- `predict.nn_fit_ds()` — extends the base method for torch `dataset`
  fits.

## Usage

``` r
# S3 method for class 'nn_fit'
predict(object, newdata = NULL, new_data = NULL, type = "response", ...)

# S3 method for class 'nn_fit_tab'
predict(object, newdata = NULL, new_data = NULL, type = "response", ...)

# S3 method for class 'nn_fit_ds'
predict(object, newdata = NULL, new_data = NULL, type = "response", ...)
```

## Arguments

- object:

  A fitted model object returned by
  [`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md).

- newdata:

  New predictor data. Accepted forms depend on the method:

  - `predict.nn_fit()`: a numeric `matrix` or coercible object.

  - `predict.nn_fit_tab()`: a `data.frame` with the same columns used
    during training; preprocessing is applied automatically via
    [`hardhat::forge()`](https://hardhat.tidymodels.org/reference/forge.html).

  - `predict.nn_fit_ds()`: a `torch` `dataset`, numeric `array`,
    `matrix`, or `data.frame`. If `NULL`, the cached fitted values from
    training are returned (not available for `type = "prob"`).

- new_data:

  Legacy alias for `newdata`. Retained for compatibility.

- type:

  Character. Output type:

  - `"response"` (default): predicted class labels (factor) for
    classification, or a numeric vector / matrix for regression.

  - `"prob"`: a numeric matrix of class probabilities (classification
    only).

- ...:

  Currently unused; reserved for future extensions.

## Value

- **Regression**: a numeric vector (single output) or matrix (multiple
  outputs).

- **Classification**, `type = "response"`: a factor with levels matching
  those seen during training.

- **Classification**, `type = "prob"`: a numeric matrix with one column
  per class, columns named by class label.

## See also

[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
