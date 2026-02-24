# Architecture specification for train_nn()

`nn_arch()` defines an architecture specification object consumed by
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
via `architecture` (or legacy `arch`).

Conceptual workflow:

1.  Define architecture with `nn_arch()`.

2.  Train with `train_nn(..., architecture = <nn_arch>)`.

3.  Predict with [`predict()`](https://rdrr.io/r/stats/predict.html).

Architecture fields mirror
[`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)
and are passed through without additional branching logic.

## Usage

``` r
nn_arch(
  nn_name = "nnModule",
  nn_layer = NULL,
  out_nn_layer = NULL,
  nn_layer_args = list(),
  layer_arg_fn = NULL,
  forward_extract = NULL,
  before_output_transform = NULL,
  after_output_transform = NULL,
  last_layer_args = list(),
  input_transform = NULL
)
```

## Arguments

- nn_name:

  Character. Name of the generated module class. Default `"nnModule"`.

- nn_layer:

  Layer type. See
  [`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md).
  Default `NULL` (`nn_linear`).

- out_nn_layer:

  Optional. Layer type forced on the last layer. Default `NULL`.

- nn_layer_args:

  Named list. Additional arguments passed to every layer constructor.
  Default [`list()`](https://rdrr.io/r/base/list.html).

- layer_arg_fn:

  Formula or function. Generates per-layer constructor arguments.
  Default `NULL` (FFNN-style: `list(in_dim, out_dim, bias = bias)`).

- forward_extract:

  Formula or function. Processes layer output in the forward pass.
  Default `NULL`.

- before_output_transform:

  Formula or function. Transforms input before the output layer. Default
  `NULL`.

- after_output_transform:

  Formula or function. Transforms output after the output layer. Default
  `NULL`.

- last_layer_args:

  Named list or formula. Extra arguments for the output layer only.
  Default [`list()`](https://rdrr.io/r/base/list.html).

- input_transform:

  Formula or function. Transforms the entire input tensor before
  training begins. Applied once to the full dataset tensor, not
  per-batch. Transforms must therefore be independent of batch size.
  Safe examples: `~ .$unsqueeze(2)` (RNN sequence dim),
  `~ .$unsqueeze(1)` (CNN channel dim). Avoid transforms that reshape
  based on `.$size(1)` as this will reflect the full dataset size, not
  the mini-batch size.

## Value

An object of class `c("nn_arch", "kindling_arch")`, implemented as a
named list of
[`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)
arguments with an `"env"` attribute capturing the calling environment
for custom layer resolution.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed()) {
    # Standard architecture object for train_nn()
    std_arch = nn_arch(nn_name = "mlp_model")

    # GRU architecture spec
    gru_arch = nn_arch(
        nn_name = "GRU",
        nn_layer = "torch::nn_gru",
        layer_arg_fn = ~ if (.is_output) {
            list(.in, .out)
        } else {
            list(input_size = .in, hidden_size = .out, batch_first = TRUE)
        },
        out_nn_layer = "torch::nn_linear",
        forward_extract = ~ .[[1]],
        before_output_transform = ~ .[, .$size(2), ],
        input_transform = ~ .$unsqueeze(2)
    )

    # Custom layer architecture (resolved from calling environment)
    custom_linear = torch::nn_module(
        "CustomLinear",
        initialize = function(in_features, out_features, bias = TRUE) {
            self$layer = torch::nn_linear(in_features, out_features, bias = bias)
        },
        forward = function(x) self$layer(x)
    )
    custom_arch = nn_arch(
        nn_name = "CustomMLP",
        nn_layer = ~ custom_linear
    )

    model = train_nn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50,
        architecture = gru_arch
    )
}
# }
```
