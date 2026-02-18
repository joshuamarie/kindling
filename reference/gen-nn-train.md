# Generalized Neural Network Trainer

**\[experimental\]**

`train_nn()` is a generic function for training neural networks with a
user-defined architecture via
[`nn_arch()`](https://kindling.joshuamarie.com/reference/nn_arch.md).
Dispatch is based on the class of `x`, allowing different preprocessing
pipelines per data type:

- `train_nn.matrix()` — raw interface, no preprocessing

- `train_nn.data.frame()` — tabular interface via
  [`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html)

- `train_nn.formula()` — formula interface via
  [`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html)

All methods delegate to the shared
[`train_nn_impl()`](https://kindling.joshuamarie.com/reference/train_nn_impl.md)
core after preprocessing. When `arch = NULL`, the model falls back to a
plain FFNN (`nn_linear`) architecture.

## Usage

``` r
train_nn(x, ...)

# S3 method for class 'matrix'
train_nn(
  x,
  y,
  hidden_neurons,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  arch = NULL,
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
  ...
)

# S3 method for class 'data.frame'
train_nn(
  x,
  y,
  hidden_neurons,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  arch = NULL,
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
  ...
)

# S3 method for class 'formula'
train_nn(
  x,
  data,
  hidden_neurons,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  arch = NULL,
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
  ...
)

# Default S3 method
train_nn(x, ...)
```

## Arguments

- x:

  Predictor data. Dispatch is based on its class:

  - `matrix`: used directly, no preprocessing

  - `data.frame`: preprocessed via
    [`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html)

  - `formula`: combined with `data` and preprocessed via
    [`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html)

- ...:

  Additional arguments passed to methods.

- y:

  Outcome data (vector, factor, or matrix). Not used when `x` is a
  formula.

- hidden_neurons:

  Integer vector. Number of neurons in each hidden layer.

- activations:

  Activation function specifications. See
  [`act_funs()`](https://kindling.joshuamarie.com/reference/act_funs.md).

- output_activation:

  Optional. Activation for the output layer.

- bias:

  Logical. Use bias weights. Default `TRUE`.

- arch:

  An
  [`nn_arch()`](https://kindling.joshuamarie.com/reference/nn_arch.md)
  object specifying the architecture. Default `NULL` (FFNN fallback).

- epochs:

  Integer. Number of training epochs. Default `100`.

- batch_size:

  Integer. Batch size for training. Default `32`.

- penalty:

  Numeric. Regularization penalty (lambda). Default `0`.

- mixture:

  Numeric. Elastic net mixing parameter (0-1). Default `0`.

- learn_rate:

  Numeric. Learning rate for optimizer. Default `0.001`.

- optimizer:

  Character. Optimizer type (`"adam"`, `"sgd"`, `"rmsprop"`). Default
  `"adam"`.

- optimizer_args:

  Named list. Additional arguments passed to the optimizer. Default
  [`list()`](https://rdrr.io/r/base/list.html).

- loss:

  Character. Loss function (`"mse"`, `"mae"`, `"cross_entropy"`,
  `"bce"`). Default `"mse"`.

- validation_split:

  Numeric. Proportion of data for validation (0-1). Default `0`.

- device:

  Character. Device to use (`"cpu"`, `"cuda"`, `"mps"`). Default `NULL`
  (auto-detect).

- verbose:

  Logical. Print training progress. Default `FALSE`.

- cache_weights:

  Logical. Cache weight matrices. Default `FALSE`.

- data:

  Data frame. Required when `x` is a formula.

## Value

An object of class `"nn_fit"`, or a subclass thereof:

- `c("nn_fit_tab", "nn_fit")` when called via `data.frame` or `formula`
  method

## Examples

``` r
# \donttest{
if (torch::torch_is_installed()) {
    # matrix method
    model = train_nn(
        x = as.matrix(iris[, 2:4]),
        y = iris$Sepal.Length,
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )

    # data.frame method
    model = train_nn(
        x = iris[, 2:4],
        y = iris$Sepal.Length,
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )

    # formula method
    model = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )
}
# }
```
