# Predict method for kindling basemodel fits

Predict method for kindling basemodel fits

## Usage

``` r
# S3 method for class 'ffnn_fit'
predict(object, newdata = NULL, new_data = NULL, type = "response", ...)

# S3 method for class 'rnn_fit'
predict(object, newdata = NULL, new_data = NULL, type = "response", ...)
```

## Arguments

- object:

  An object of class `"ffnn_fit"` or `"rnn_fit"`.

- newdata:

  Data frame. New data for predictions. If `NULL`, uses the original
  training data (if available).

- new_data:

  Alternative to newdata (for consistency with hardhat).

- type:

  Character. Type of prediction:

  - `"response"` (default) – predicted values or predicted classes

  - `"prob"` – class probabilities (only for classification models)

- ...:

  Currently unused.

## Value

- For **regression** models: a numeric vector (single output) or matrix
  (multiple outputs) of predicted values.

- For **classification** models:

  - `type = "response"`: a factor vector of predicted class labels

  - `type = "prob"`: a numeric matrix of class probabilities, with
    columns named after the class levels.
