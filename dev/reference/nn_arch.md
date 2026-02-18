# Architecture specification for train_nn()

`nn_arch()` is a helper that bundles
[`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)
arguments into a single object passed to
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
via the `arch` parameter. All arguments mirror those of
[`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)
exactly, including their defaults.

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
  input_transform = NULL,
  use_namespace = TRUE
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

  Formula or function. Transforms the input tensor before it enters the
  model. Applied to all tensors (train, validation, inference). Useful
  for architectures that require a specific input shape, e.g. RNNs
  needing a sequence dimension: `~ .$unsqueeze(2)`. Default `NULL`.

- use_namespace:

  Logical or character. Controls torch namespace prefixing. Default
  `TRUE`.

## Value

An object of class `"nn_arch"`, a named list of
[`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)
arguments.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed()) {
    # GRU architecture spec
    gru_arch = nn_arch(
        nn_name = "GRU",
        nn_layer = "nn_gru",
        layer_arg_fn = ~ if (.is_output) {
            list(.in, .out)
        } else {
            list(input_size = .in, hidden_size = .out, batch_first = TRUE)
        },
        out_nn_layer = "nn_linear",
        forward_extract = ~ .[[1]],
        before_output_transform = ~ .[, .$size(2), ],
        input_transform = ~ .$unsqueeze(2)
    )

    model = train_nn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50,
        arch = gru_arch
    )
}
# }
```
