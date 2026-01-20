# Recurrent Neural Network via kindling

`rnn_kindling()` defines a recurrent neural network model that can be
used for classification or regression on sequential data. It integrates
with the tidymodels ecosystem and uses the torch backend via kindling.

## Usage

``` r
rnn_kindling(
  mode = "unknown",
  engine = "kindling",
  hidden_neurons = NULL,
  rnn_type = NULL,
  activations = NULL,
  output_activation = NULL,
  bias = NULL,
  bidirectional = NULL,
  dropout = NULL,
  epochs = NULL,
  batch_size = NULL,
  penalty = NULL,
  mixture = NULL,
  learn_rate = NULL,
  optimizer = NULL,
  loss = NULL,
  validation_split = NULL,
  device = NULL,
  verbose = NULL
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

- rnn_type:

  A character string for the type of RNN cell ("rnn", "lstm", "gru").
  Can be tuned.

- activations:

  A character vector of activation function names for each hidden layer
  (e.g., "relu", "tanh", "sigmoid"). Can be tuned.

- output_activation:

  A character string for the output activation function. Can be tuned.

- bias:

  Logical for whether to include bias terms. Can be tuned.

- bidirectional:

  A logical indicating whether to use bidirectional RNN. Can be tuned.

- dropout:

  A number between 0 and 1 for dropout rate between layers. Can be
  tuned.

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

- loss:

  A character string for the loss function ("mse", "mae",
  "cross_entropy", "bce"). Can be tuned.

- validation_split:

  A number between 0 and 1 for the proportion of data used for
  validation. Can be tuned.

- device:

  A character string for the device to use ("cpu", "cuda", "mps"). If
  NULL, auto-detects available GPU. Can be tuned.

- verbose:

  Logical for whether to print training progress. Default FALSE.

## Value

A model specification object with class `rnn_kindling`.

## Details

This function creates a model specification for a recurrent neural
network that can be used within tidymodels workflows. The model
supports:

- Multiple RNN types: basic RNN, LSTM, and GRU

- Bidirectional processing

- Dropout regularization

- GPU acceleration (CUDA, MPS, or CPU)

- Hyperparameter tuning integration

- Both regression and classification tasks

The `device` parameter controls where computation occurs:

- `NULL` (default): Auto-detect best available device (CUDA \> MPS \>
  CPU)

- `"cuda"`: Use NVIDIA GPU

- `"mps"`: Use Apple Silicon GPU

- `"cpu"`: Use CPU only

## Examples

``` r
if (FALSE) { # \dontrun{
if (torch::torch_is_installed()) {
    box::use(
        recipes[recipe],
        workflows[workflow, add_recipe, add_model],
        parsnip[fit]
    )

    # Model specs
    rnn_spec = rnn_kindling(
        mode = "classification",
        hidden_neurons = c(64, 32),
        rnn_type = "lstm",
        activation = c("relu", "elu"),
        epochs = 100,
        bidirectional = TRUE
    )

    wf = workflow() |>
        add_recipe(recipe(Species ~ ., data = iris)) |>
        add_model(rnn_spec)

    fit_wf = fit(wf, data = iris)
    fit_wf
} else {
    message("Torch not fully installed — skipping example")
}
} # }
```
