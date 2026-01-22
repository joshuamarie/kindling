# Predict Method for FFNN Fits

Predict Method for FFNN Fits

## Usage

``` r
# S3 method for class 'ffnn_fit'
predict(object, newdata = NULL, type = "response", ...)
```

## Arguments

- object:

  An object of class "ffnn_fit".

- newdata:

  Data frame. New data for predictions.

- type:

  Character. Type of prediction: "response" (default) or "prob" for
  classification.

- ...:

  Additional arguments (unused).

## Value

For regression: A numeric vector or matrix of predictions. For
classification with type = "response": A factor vector of predicted
classes. For classification with type = "prob": A numeric matrix of
class probabilities with columns named by class levels.
