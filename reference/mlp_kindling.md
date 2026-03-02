# Multi-Layer Perceptron (Feedforward Neural Network) via kindling

`mlp_kindling()` defines a feedforward neural network model that can be
used for classification or regression. It integrates with the tidymodels
ecosystem and uses the torch backend via kindling.

## Usage

``` r
mlp_kindling(
  mode = "unknown",
  engine = "kindling",
  hidden_neurons = NULL,
  activations = NULL,
  output_activation = NULL,
  bias = NULL,
  epochs = NULL,
  batch_size = NULL,
  penalty = NULL,
  mixture = NULL,
  learn_rate = NULL,
  optimizer = NULL,
  validation_split = NULL,
  optimizer_args = NULL,
  loss = NULL,
  architecture = NULL,
  flatten_input = NULL,
  early_stopping = NULL,
  device = NULL,
  verbose = NULL,
  cache_weights = NULL
)
```

## Arguments

- mode:

  A single character string for the type of model. Possible values are
  "unknown", "regression", or "classification".

- engine:

  A single character string specifying what computational engine to use
  for fitting. Currently only "kindling" is supported.

- hidden_neurons:

  An integer vector for the number of units in each hidden layer. Can be
  tuned.

- activations:

  A character vector of activation function names for each hidden layer
  (e.g., "relu", "tanh", "sigmoid"). Can be tuned.

- output_activation:

  A character string for the output activation function. Can be tuned.

- bias:

  Logical for whether to include bias terms. Can be tuned.

- epochs:

  An integer for the number of training iterations. Can be tuned.

- batch_size:

  An integer for the batch size during training. Can be tuned.

- penalty:

  A number for the regularization penalty (lambda). Default `0` (no
  regularization). Higher values increase regularization strength. Can
  be tuned.

- mixture:

  A number between 0 and 1 for the elastic net mixing parameter. Default
  `0` (pure L2/Ridge regularization).

  - `0`: Pure L2 regularization (Ridge)

  - `1`: Pure L1 regularization (Lasso)

  - `0 < mixture < 1`: Elastic net (combination of L1 and L2) Only
    relevant when `penalty > 0`. Can be tuned.

- learn_rate:

  A number for the learning rate. Can be tuned.

- optimizer:

  A character string for the optimizer type ("adam", "sgd", "rmsprop").
  Can be tuned.

- validation_split:

  A number between 0 and 1 for the proportion of data used for
  validation. Can be tuned.

- optimizer_args:

  A named list of additional arguments passed to the optimizer. Cannot
  be tuned — pass via `set_engine()`.

- loss:

  A character string for the loss function ("mse", "mae",
  "cross_entropy", "bce"). Cannot be tuned — pass via `set_engine()`.

- architecture:

  An
  [`nn_arch()`](https://kindling.joshuamarie.com/reference/nn_arch.md)
  object for a custom architecture. Cannot be tuned — pass via
  `set_engine()`.

- flatten_input:

  Logical or `NULL`. Controls input flattening. Cannot be tuned — pass
  via `set_engine()`.

- early_stopping:

  An
  [`early_stop()`](https://kindling.joshuamarie.com/reference/early_stop.md)
  object or `NULL`. Cannot be tuned — pass via `set_engine()`.

- device:

  A character string for the device ("cpu", "cuda", "mps"). Cannot be
  tuned — pass via `set_engine()`.

- verbose:

  Logical for whether to print training progress. Cannot be tuned — pass
  via `set_engine()`.

- cache_weights:

  Logical. If `TRUE`, stores trained weight matrices in the returned
  object. Cannot be tuned — pass via `set_engine()`.

## Value

A model specification object with class `mlp_kindling`.

## Details

This function creates a model specification for a feedforward neural
network that can be used within tidymodels workflows. The model
supports:

- Multiple hidden layers with configurable units

- Various activation functions per layer

- GPU acceleration (CUDA, MPS, or CPU)

- Hyperparameter tuning integration

- Both regression and classification tasks

Parameters that cannot be tuned (`architecture`, `flatten_input`,
`early_stopping`, `device`, `verbose`, `cache_weights`,
`optimizer_args`, `loss`) must be set via `set_engine()`, not as
arguments to `mlp_kindling()`.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed()) {
    box::use(
        recipes[recipe],
        workflows[workflow, add_recipe, add_model],
        tune[tune],
        parsnip[fit]
    )

    # library(recipes)
    # library(workflows)
    # library(parsnip)
    # library(tune)

    # Model specs
    mlp_spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = c(128, 64, 32),
        activation = c("relu", "relu", "relu"),
        epochs = 100
    )

    # If you want to tune
    mlp_tune_spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = tune(),
        activation = tune(),
        epochs = tune(),
        learn_rate = tune()
    )
     wf = workflow() |>
        add_recipe(recipe(Species ~ ., data = iris)) |>
        add_model(mlp_spec)

     fit_wf = fit(wf, data = iris)
} else {
    message("Torch not fully installed — skipping example")
}
# }
```
