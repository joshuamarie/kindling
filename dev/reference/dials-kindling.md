# Tunable hyperparameters for `kindling` models

These parameters extend the **dials** framework to support
hyperparameter tuning of neural networks built with the `{kindling}`
package. They control network architecture, activation functions,
optimization, and training behavior.

## Usage

``` r
n_hlayers(range = c(1L, 2L), trans = NULL)

hidden_neurons(range = c(8L, 512L), disc_values = NULL, trans = NULL)

activations(
  values = c("relu", "relu6", "elu", "selu", "celu", "leaky_relu", "gelu", "softplus",
    "softshrink", "softsign", "tanhshrink", "hardtanh", "hardshrink", "hardswish",
    "hardsigmoid", "silu", "mish", "logsigmoid")
)

output_activation(
  values = c("relu", "elu", "selu", "softplus", "softmax", "log_softmax", "logsigmoid",
    "hardtanh", "hardsigmoid", "silu")
)

optimizer(values = c("adam", "sgd", "rmsprop", "adamw"))

bias(values = c(TRUE, FALSE))

validation_split(range = c(0, 0.5), trans = NULL)

bidirectional(values = c(TRUE, FALSE))
```

## Arguments

- range:

  A two-element numeric vector with the default lower and upper bounds.

- trans:

  An optional transformation; `NULL` for none.

- disc_values:

  `NULL` (default) or an integer vector of specific possible disc_values
  (e.g., `c(32L, 64L, 128L, 256L)`). When provided, tuning will be
  restricted to these discrete values. The range is automatically
  derived from these values if not explicitly given. The `trans`
  parameter would still be ignored by this parameter when supplied.

- values:

  Logical vector of possible values.

## Value

Each function returns a `dials` parameter object:

- `n_hlayers()`:

  A quantitative parameter for the number of hidden layers

- `hidden_neurons()`:

  A quantitative parameter for hidden units per layer

- `activations()`:

  A qualitative parameter for activation function names

- `output_activation()`:

  A qualitative parameter for output activation

- `optimizer()`:

  A qualitative parameter for optimizer type

- `bias()`:

  A qualitative parameter for bias inclusion

- `validation_split()`:

  A quantitative parameter for validation proportion

- `bidirectional()`:

  A qualitative parameter for bidirectional RNN

## Architecture Strategy

Since tidymodels tuning works with independent parameters, we use a
simplified approach where:

- `hidden_neurons` specifies a single value that will be used for ALL
  layers

- `activations` specifies a single activation that will be used for ALL
  layers

- `n_hlayers` controls the depth

For more complex architectures with different neurons/activations per
layer, users should manually specify these after tuning or use custom
tuning logic.

## Parameters

- `n_hlayers`:

  Number of hidden layers in the network.

- `hidden_neurons`:

  Number of units per hidden layer (applied to all layers).

- `activation`:

  Single activation function applied to all hidden layers.

- `output_activation`:

  Activation function for the output layer.

- `optimizer`:

  Optimizer algorithm.

- `bias`:

  Whether to include bias terms in layers.

- `validation_split`:

  Proportion of training data held out for validation.

- `bidirectional`:

  Whether RNN layers are bidirectional.

## Number of Hidden Layers

Controls the depth of the network. When tuning, this will determine how
many layers are created, each with `hidden_neurons` units and
`activations` function.

## Hidden Units per Layer

Specifies the number of units per hidden layer.

## Activation Function (Hidden Layers)

Activation functions for hidden layers.

## Output Activation Function

Activation function applied to the output layer. Values must correspond
to `torch::nnf_*` functions.

## Optimizer Type

The optimization algorithm used during training.

## Include Bias Terms

Whether layers should include bias parameters.

## Validation Split Proportion

Fraction of the training data to use as a validation set during
training.

## Bidirectional RNN

Whether recurrent layers should process sequences in both directions.

## Examples

``` r
# \donttest{
library(dials)
#> Loading required package: scales
library(tune)

# Create a tuning grid
grid = grid_regular(
    n_hlayers(range = c(1L, 4L)),
    hidden_neurons(range = c(32L, 256L)),
    activations(c('relu', 'elu', 'selu')),
    levels = c(4, 5, 3)
)

# Use in a model spec
mlp_spec = mlp_kindling(
    mode = "classification",
    hidden_neurons = tune(),
    activations = tune(),
    epochs = tune(),
    learn_rate = tune()
)
# }
```
