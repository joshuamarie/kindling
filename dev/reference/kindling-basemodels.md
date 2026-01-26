# Base models for Neural Network Training in kindling

Base models for Neural Network Training in kindling

## Usage

``` r
ffnn(
  formula = NULL,
  data = NULL,
  hidden_neurons,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  epochs = 100,
  batch_size = 32,
  penalty = 0,
  mixture = 0,
  learn_rate = 0.001,
  optimizer = "adam",
  optimizer_args = list(),
  loss = "mse",
  validation_split = 0,
  device = NULL,
  verbose = FALSE,
  cache_weights = FALSE,
  ...,
  x = NULL,
  y = NULL
)

rnn(
  formula = NULL,
  data = NULL,
  hidden_neurons,
  rnn_type = "lstm",
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  bidirectional = TRUE,
  dropout = 0,
  epochs = 100,
  batch_size = 32,
  penalty = 0,
  mixture = 0,
  learn_rate = 0.001,
  optimizer = "adam",
  optimizer_args = list(),
  loss = "mse",
  validation_split = 0,
  device = NULL,
  verbose = FALSE,
  cache_weights = FALSE,
  ...,
  x = NULL,
  y = NULL
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
  [`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md).

- output_activation:

  Optional. Activation for output layer.

- bias:

  Logical. Use bias weights. Default `TRUE`.

- epochs:

  Integer. Number of training epochs. Default `100`.

- batch_size:

  Integer. Batch size for training. Default `32`.

- penalty:

  Numeric. Regularization penalty (lambda). Default `0` (no
  regularization).

- mixture:

  Numeric. Elastic net mixing parameter (0-1). Default `0`.

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

  Logical. Cache weight matrices for faster variable importance. Default
  `FALSE`.

- ...:

  Additional arguments. Can be used to pass `x` and `y` for direct
  interface.

- x:

  When not using formula: predictor data (data.frame or matrix).

- y:

  When not using formula: outcome data (vector, factor, or matrix).

- rnn_type:

  Character. Type of RNN ("rnn", "lstm", "gru"). Default `"lstm"`.

- bidirectional:

  Logical. Use bidirectional RNN. Default `TRUE`.

- dropout:

  Numeric. Dropout rate between layers. Default `0`.

## Value

An object of class "ffnn_fit" containing the trained model and metadata.

## FFNN

Train a feed-forward neural network using the torch package.

## RNN

Train a recurrent neural network using the torch package.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed()) {
    # Formula interface (original)
    model_reg = ffnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )

    # XY interface (new)
    model_xy = ffnn(
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50,
        x = iris[, 2:4],
        y = iris$Sepal.Length
    )
}
# }

# \donttest{
if (torch::torch_is_installed()) {
    # Formula interface (original)
    model_rnn = rnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        rnn_type = "lstm",
        activations = "relu",
        epochs = 50
    )

    # XY interface (new)
    model_xy = rnn(
        hidden_neurons = c(64, 32),
        rnn_type = "gru",
        epochs = 50,
        x = iris[, 2:4],
        y = iris$Sepal.Length
    )
}
# }
```
