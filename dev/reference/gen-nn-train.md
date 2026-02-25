# Generalized Neural Network Trainer

**\[experimental\]**

`train_nn()` is a generic function for training neural networks with a
user-defined architecture via
[`nn_arch()`](https://kindling.joshuamarie.com/dev/reference/nn_arch.md).
Dispatch is based on the class of `x`.

Recommended workflow:

1.  Define architecture with
    [`nn_arch()`](https://kindling.joshuamarie.com/dev/reference/nn_arch.md)
    (optional).

2.  Train with `train_nn()`.

3.  Predict with
    [`predict.nn_fit()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-predict.md).

All methods delegate to a shared implementation core after
preprocessing. When `architecture = NULL`, the model falls back to a
plain feed-forward neural network (`nn_linear`) architecture.

## Usage

``` r
train_nn(x, ...)

# S3 method for class 'matrix'
train_nn(
  x,
  y,
  hidden_neurons = NULL,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  arch = NULL,
  architecture = NULL,
  early_stopping = NULL,
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
  hidden_neurons = NULL,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  arch = NULL,
  architecture = NULL,
  early_stopping = NULL,
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
  hidden_neurons = NULL,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  arch = NULL,
  architecture = NULL,
  early_stopping = NULL,
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

# S3 method for class 'dataset'
train_nn(
  x,
  y = NULL,
  hidden_neurons = NULL,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  arch = NULL,
  architecture = NULL,
  flatten_input = NULL,
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
  n_classes = NULL,
  ...
)
```

## Arguments

- x:

  Dispatch is based on its current class:

  - `matrix`: used directly, no preprocessing applied.

  - `data.frame`: preprocessed via
    [`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html).
    `y` may be a vector / factor / matrix of outcomes, or a formula
    describing the outcome–predictor relationship within `x`.

  - `formula`: combined with `data` and preprocessed via
    [`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html).

  - `dataset`: a `torch` dataset object; batched via
    [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html).
    This is the recommended interface for sequence/time-series and image
    data.

- ...:

  Additional arguments passed to specific methods.

- y:

  Outcome data. Interpretation depends on the method:

  - For the `matrix` and `data.frame` methods: a numeric vector, factor,
    or matrix of outcomes.

  - For the `data.frame` method only: alternatively a formula of the
    form `outcome ~ predictors`, evaluated against `x`.

  - Ignored when `x` is a formula (outcome is taken from the formula) or
    a `dataset` (labels come from the dataset itself).

- hidden_neurons:

  Integer vector specifying the number of neurons in each hidden layer,
  e.g. `c(128, 64)` for two hidden layers. When `NULL` or missing, no
  hidden layers are used and the model reduces to a single linear
  mapping from inputs to outputs.

- activations:

  Activation function specification(s) for the hidden layers. See
  [`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md)
  for supported values. Recycled if a single value is given.

- output_activation:

  Optional activation function for the output layer. Defaults to `NULL`
  (no activation / linear output).

- bias:

  Logical. Whether to include bias terms in each layer. Default `TRUE`.

- arch:

  Backward-compatible alias for `architecture`. If both are supplied,
  they must be identical.

- architecture:

  An
  [`nn_arch()`](https://kindling.joshuamarie.com/dev/reference/nn_arch.md)
  object specifying a custom architecture. Default `NULL`, which falls
  back to a standard feed-forward network.

- early_stopping:

  An
  [`early_stop()`](https://kindling.joshuamarie.com/dev/reference/early_stop.md)
  object specifying early stopping behaviour, or `NULL` (default) to
  disable early stopping. When supplied, training halts if the monitored
  metric does not improve by at least `min_delta` for `patience`
  consecutive epochs. Example:
  `early_stopping = early_stop(patience = 10)`.

- epochs:

  Positive integer. Number of full passes over the training data.
  Default `100`.

- batch_size:

  Positive integer. Number of samples per mini-batch. Default `32`.

- penalty:

  Non-negative numeric. L1/L2 regularization strength (lambda). Default
  `0` (no regularization).

- mixture:

  Numeric in \[0, 1\]. Elastic net mixing parameter: `0` = pure ridge
  (L2), `1` = pure lasso (L1). Default `0`.

- learn_rate:

  Positive numeric. Step size for the optimizer. Default `0.001`.

- optimizer:

  Character. Optimizer algorithm. One of `"adam"` (default), `"sgd"`, or
  `"rmsprop"`.

- optimizer_args:

  Named list of additional arguments forwarded to the optimizer
  constructor (e.g. `list(momentum = 0.9)` for SGD). Default
  [`list()`](https://rdrr.io/r/base/list.html).

- loss:

  Character or function. Loss function used during training. Built-in
  options: `"mse"` (default), `"mae"`, `"cross_entropy"`, or `"bce"`.
  For classification tasks with a scalar label, `"cross_entropy"` is set
  automatically. Alternatively, supply a custom function or formula with
  signature `function(input, target)` returning a scalar `torch_tensor`.

- validation_split:

  Numeric in \[0, 1). Proportion of training data held out for
  validation. Default `0` (no validation set).

- device:

  Character. Compute device: `"cpu"`, `"cuda"`, or `"mps"`. Default
  `NULL`, which auto-detects the best available device.

- verbose:

  Logical. If `TRUE`, prints loss (and validation loss) at regular
  intervals during training. Default `FALSE`.

- cache_weights:

  Logical. If `TRUE`, stores a copy of the trained weight matrices in
  the returned object under `$cached_weights`. Default `FALSE`.

- data:

  A data frame. Required when `x` is a formula.

- flatten_input:

  Logical or `NULL` (dataset method only). Controls whether each
  batch/sample is flattened to 2D before entering the model. `NULL`
  (default) auto-selects: `TRUE` when `architecture = NULL`, otherwise
  `FALSE`.

- n_classes:

  Positive integer. Number of output classes. Required when `x` is a
  `dataset` with scalar (classification) labels; ignored otherwise.

## Value

An object of class `"nn_fit"`, or one of its subclasses:

- `c("nn_fit_tab", "nn_fit")` — returned by the `data.frame` and
  `formula` methods

- `c("nn_fit_ds", "nn_fit")` — returned by the `dataset` method

All subclasses share a common structure. See **Details** for the list of
components.

## Details

The returned `"nn_fit"` object is a named list with the following
components:

- `model` — the trained
  [`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
  object

- `fitted` — fitted values on the training data (or `NULL` for dataset
  fits)

- `loss_history` — numeric vector of per-epoch training loss, trimmed to
  actual epochs run (relevant when early stopping is active)

- `val_loss_history` — per-epoch validation loss, or `NULL` if
  `validation_split = 0`

- `n_epochs` — number of epochs actually trained

- `stopped_epoch` — epoch at which early stopping triggered, or `NA` if
  training ran to completion

- `hidden_neurons`, `activations`, `output_activation` — architecture
  spec

- `penalty`, `mixture` — regularization settings

- `feature_names`, `response_name` — variable names (tabular methods
  only)

- `no_x`, `no_y` — number of input features and output nodes

- `is_classification` — logical flag

- `y_levels`, `n_classes` — class labels and count (classification only)

- `device` — device the model is on

- `cached_weights` — list of weight matrices, or `NULL`

- `arch` — the `nn_arch` object used, or `NULL`

## Supported tasks and input formats

`train_nn()` is task-agnostic by design (no explicit `task` argument).
Task behavior is determined by your input interface and architecture:

- **Tabular data**: use `matrix`, `data.frame`, or `formula` methods.

- **Time series**: use the `dataset` method with per-item tensors shaped
  as `[time, features]` (or your preferred convention) and a recurrent
  architecture via
  [`nn_arch()`](https://kindling.joshuamarie.com/dev/reference/nn_arch.md).

- **Image classification**: use the `dataset` method with per-item
  tensors shaped for your first layer (commonly
  `[channels, height, width]` for
  [`torch::nn_conv2d`](https://torch.mlverse.org/docs/reference/nn_conv2d.html)).
  If your source arrays are channel-last, reorder in the dataset or via
  `input_transform`.

## Matrix method

When `x` is supplied as a raw numeric matrix, no preprocessing is
applied. Data is passed directly to the shared `train_nn_impl` core.

## Data frame method

When `x` is a data frame, `y` can be either a vector / factor / matrix
of outcomes, or a formula of the form `outcome ~ predictors` evaluated
against `x`. Preprocessing is handled by
[`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html).

## Formula method

When `x` is a formula, `data` must be supplied as the data frame against
which the formula is evaluated. Preprocessing is handled by
[`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html).

## Dataset method (`train_nn.dataset()`)

Trains a neural network directly on a `torch` dataset object. Batching
and lazy loading are handled by
[`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html),
making this method well-suited for large datasets that do not fit
entirely in memory.

Architecture configuration follows the same contract as other
`train_nn()` methods via `architecture = nn_arch(...)` (or legacy
`arch = ...`). For non-tabular inputs (time series, images), set
`flatten_input = FALSE` to preserve tensor dimensions expected by
recurrent or convolutional layers.

Labels are taken from the second element of each dataset item (i.e.
`dataset[[i]][[2]]`), so `y` is ignored. When the label is a scalar
tensor, a classification task is assumed and `n_classes` must be
supplied. The loss is automatically switched to `"cross_entropy"` in
that case.

Fitted values are **not** cached in the returned object. Use
[`predict.nn_fit_ds()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-predict.md)
with `newdata` to obtain predictions after training.

## See also

[`predict.nn_fit()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-predict.md),
[`nn_arch()`](https://kindling.joshuamarie.com/dev/reference/nn_arch.md),
[`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md),
[`early_stop()`](https://kindling.joshuamarie.com/dev/reference/early_stop.md)

## Examples

``` r
# \donttest{
if (torch::torch_is_installed()) {
    # Matrix method — no preprocessing
    model = train_nn(
        x = as.matrix(iris[, 2:4]),
        y = iris$Sepal.Length,
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )

    # Data frame method — y as a vector
    model = train_nn(
        x = iris[, 2:4],
        y = iris$Sepal.Length,
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )

    # Data frame method — y as a formula evaluated against x
    model = train_nn(
        x = iris,
        y = Sepal.Length ~ . - Species,
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )

    # Formula method — outcome derived from formula
    model = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )

    # No hidden layers — linear model
    model = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        epochs = 50
    )

    # Architecture object (nn_arch -> train_nn)
    mlp_arch = nn_arch(nn_name = "mlp_model")
    model = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        activations = "relu",
        architecture = mlp_arch,
        epochs = 50
    )

    # Custom layer architecture
    custom_linear = torch::nn_module(
        "CustomLinear",
        initialize = function(in_features, out_features, bias = TRUE) {
            self$layer = torch::nn_linear(in_features, out_features, bias = bias)
        },
        forward = function(x) self$layer(x)
    )
    custom_arch = nn_arch(
        nn_name = "custom_linear_mlp",
        nn_layer = ~ custom_linear
    )
    model = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(16, 8),
        activations = "relu",
        architecture = custom_arch,
        epochs = 50
    )

    # With early stopping
    model = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 200,
        validation_split = 0.2,
        early_stopping = early_stop(patience = 10)
    )
}
# }

# \donttest{
if (torch::torch_is_installed()) {
    # torch dataset method — labels come from the dataset itself
    iris_cls_dataset = torch::dataset(
        name = "iris_cls_dataset",
        
        initialize = function(data = iris) {
            self$x = torch::torch_tensor(
                as.matrix(data[, 1:4]),
                dtype = torch::torch_float32()
            )
            # Species is a factor; convert to integer (1-indexed -> keep as-is for cross_entropy)
            self$y = torch::torch_tensor(
                as.integer(data$Species),
                dtype = torch::torch_long()
            )
        },
        
        .getitem = function(i) {
            list(self$x[i, ], self$y[i])
        },
        
        .length = function() {
            self$x$size(1)
        }
    )()
    
    model_nn_ds = train_nn(
        x = iris_cls_dataset,
        hidden_neurons = c(32, 10),
        activations = "relu",
        epochs = 80,
        batch_size = 16,
        learn_rate = 0.01,
        n_classes = 3, # Iris dataset has only 3 species
        validation_split = 0.2,
        verbose = TRUE
    )
    
    pred_nn = predict(model_nn_ds, iris_cls_dataset)
    class_preds = c("Setosa", "Versicolor", "Virginica")[predict(model_nn_ds, iris_cls_dataset)]
    
    # Confusion Matrix
    table(actual = iris$Species, pred = class_preds)
}
#> → Auto-detected classification task. Using cross_entropy loss.
#> ℹ Using device: cpu
#> Epoch 8/80 - Loss: 0.1534 - Val Loss: 0.1958
#> Epoch 16/80 - Loss: 0.1935 - Val Loss: 0.3148
#> Epoch 24/80 - Loss: 0.0713 - Val Loss: 0.1177
#> Epoch 32/80 - Loss: 0.0564 - Val Loss: 0.1056
#> Epoch 40/80 - Loss: 0.0686 - Val Loss: 0.1016
#> Epoch 48/80 - Loss: 0.0807 - Val Loss: 0.0997
#> Epoch 56/80 - Loss: 0.0564 - Val Loss: 0.1116
#> Epoch 64/80 - Loss: 0.0572 - Val Loss: 0.1159
#> Epoch 72/80 - Loss: 0.0987 - Val Loss: 0.1011
#> Epoch 80/80 - Loss: 0.0704 - Val Loss: 0.1077
#>             pred
#> actual       Setosa Versicolor Virginica
#>   setosa         50          0         0
#>   versicolor      0         49         1
#>   virginica       0          2        48
# }
```
