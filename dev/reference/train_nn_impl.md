# Shared core implementation

Shared core implementation

## Usage

``` r
train_nn_impl(
  x,
  y,
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
  arch = NULL,
  fit_class = "nn_fit"
)
```
