# Base models for Neural Network Training in kindling

Base models for Neural Network Training in kindling

## Usage

``` r
ffnn(
  formula,
  data,
  hidden_neurons,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  epochs = 100,
  batch_size = 32,
  learn_rate = 0.001,
  optimizer = "adam",
  optimizer_args = list(),
  loss = "mse",
  validation_split = 0,
  device = NULL,
  verbose = FALSE,
  cache_weights = FALSE,
  ...
)

rnn(
  formula,
  data,
  hidden_neurons,
  rnn_type = "lstm",
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  bidirectional = TRUE,
  dropout = 0,
  epochs = 100,
  batch_size = 32,
  learn_rate = 0.001,
  optimizer = "adam",
  optimizer_args = list(),
  loss = "mse",
  validation_split = 0,
  device = NULL,
  verbose = FALSE,
  cache_weights = FALSE,
  ...
)
```

## Arguments

- formula:

  Formula. Model formula (e.g., y ~ x1 + x2).

- data:

  Data frame. Training data.

- hidden_neurons:

  Integer vector. Number of neurons in each hidden layer.

- activations:

  Activation function specifications. See
  [`act_funs()`](https://kindling.joshuamarie.com/reference/act_funs.md).

- output_activation:

  Optional. Activation for output layer.

- bias:

  Logical. Use bias weights. Default `TRUE`.

- epochs:

  Integer. Number of training epochs. Default `100`.

- batch_size:

  Integer. Batch size for training. Default `32`.

- learn_rate:

  Numeric. Learning rate for optimizer. Default `0.001`.

- optimizer:

  Character. Optimizer type ("adam", "sgd", "rmsprop"). Default
  `"adam"`.

- optimizer_args:

  Named list. Additional arguments passed to the optimizer. Default
  [`list()`](https://rdrr.io/r/base/list.html).

- loss:

  Character. Loss function ("mse", "mae", "cross_entropy", "bce").
  Default `"mse"`.

- validation_split:

  Numeric. Proportion of data for validation (0-1). Default `0`.

- device:

  Character. Device to use ("cpu", "cuda", "mps"). Default `NULL`
  (auto-detect).

- verbose:

  Logical. Print training progress. Default `FALSE`.

- cache_weights:

  Logical. Cache weight matrices for faster variable importance
  computation. Default `FALSE`. When `TRUE`, weight matrices are
  extracted and stored in the returned object, avoiding repeated
  extraction during importance calculations. Only enable if you plan to
  compute variable importance multiple times.

- ...:

  Not used. Reserved for future extensions.

- rnn_type:

  Character. Type of RNN ("rnn", "lstm", "gru"). Default `"lstm"`.

- bidirectional:

  Logical. Use bidirectional RNN. Default `TRUE`.

- dropout:

  Numeric. Dropout rate between layers. Default `0`.

## Value

An object of class "ffnn_fit" containing:

- model:

  Trained torch module

- formula:

  Model formula

- fitted.values:

  Fitted values on training data

- loss_history:

  Training loss per epoch

- val_loss_history:

  Validation loss per epoch (if validation_split \> 0)

- n_epochs:

  Number of epochs trained

- feature_names:

  Names of predictor variables

- response_name:

  Name of response variable

- device:

  Device used for training

- cached_weights:

  Weight matrices (only if cache_weights = TRUE)

## FFNN

Train a feed-forward neural network using the torch package.

## RNN

Train a recurrent neural network using the torch package.

## Examples

``` r
if (FALSE) { # \dontrun{
if (torch::torch_is_installed()) {
    # Regression task (auto-detect GPU)
    model_reg = ffnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50,
        verbose = FALSE
    )

    # With weight caching for multiple importance calculations
    model_cached = ffnn(
        Species ~ .,
        data = iris,
        hidden_neurons = c(128, 64, 32),
        activations = "relu",
        cache_weights = TRUE,
        epochs = 100
    )
} else {
    message("Torch not fully installed – skipping example")
}

} # }

if (FALSE) { # \dontrun{
# Regression with LSTM on GPU
if (torch::torch_is_installed()) {
    model_rnn = rnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        rnn_type = "lstm",
        activations = "relu",
        epochs = 50
    )

    # With weight caching
    model_cached = rnn(
        Species ~ .,
        data = iris,
        hidden_neurons = c(128, 64),
        rnn_type = "gru",
        cache_weights = TRUE,
        epochs = 100
    )
} else {
    message("Torch not fully installed – skipping example")
}
} # }
```
