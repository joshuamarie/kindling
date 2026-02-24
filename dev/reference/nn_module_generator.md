# Generalized Neural Network Module Expression Generator

**\[experimental\]**

`nn_module_generator()` is a generalized function that generates neural
network module expressions for various architectures. It provides a
flexible framework for creating custom neural network modules by
parameterizing layer types, construction arguments, and forward pass
behavior.

While designed primarily for `{torch}` modules, it can work with custom
layer implementations from the current environment, including
user-defined layers like RBF networks, custom attention mechanisms, or
other novel architectures.

This function serves as the foundation for specialized generators like
[`ffnn_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_gens.md)
and
[`rnn_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_gens.md),
but can be used directly to create custom architectures.

## Usage

``` r
nn_module_generator(
  nn_name = "nnModule",
  nn_layer = NULL,
  out_nn_layer = NULL,
  nn_layer_args = list(),
  layer_arg_fn = NULL,
  forward_extract = NULL,
  before_output_transform = NULL,
  after_output_transform = NULL,
  last_layer_args = list(),
  hd_neurons,
  no_x,
  no_y,
  activations = NULL,
  output_activation = NULL,
  bias = TRUE,
  eval = FALSE,
  .env = parent.frame(),
  ...
)
```

## Arguments

- nn_name:

  Character string specifying the name of the generated neural network
  module class. Default is `"nnModule"`.

- nn_layer:

  The type of neural network layer to use. Can be specified as:

  - `NULL` (default): Uses `nn_linear()` from `{torch}`

  - Character string: e.g., `"nn_linear"`, `"nn_gru"`, `"nn_lstm"`,
    `"some_custom_layer"`

  - Named function: A function object that constructs the layer

  - Anonymous function: e.g., `\() nn_linear()` or
    `function() nn_linear()`

  The layer constructor is first searched in the current environment,
  then in parent environments, and finally falls back to the `{torch}`
  namespace. This allows you to use custom layer implementations
  alongside standard torch layers.

- out_nn_layer:

  Default `NULL`. If supplied, it forces to be the neural network layer
  to be used on the last layer. Can be specified as:

  - Character string, e.g. `"nn_linear"`, `"nn_gru"`, `"nn_lstm"`,
    `"some_custom_layer"`

  - Named function: A function object that constructs the layer

  - Formula interface, e.g. `~torch::nn_linear`, `~some_custom_layer`

  Internally, it almost works the same as `nn_layer` parameter.

- nn_layer_args:

  Named list of additional arguments passed to the layer constructor
  specified by `nn_layer`. These arguments are applied to all layers.
  For layer-specific arguments, use `layer_arg_fn`. Default is an empty
  list.

- layer_arg_fn:

  Optional function or formula that generates layer-specific
  construction arguments. Can be specified as:

  - **Formula**: `~ list(input_size = .in, hidden_size = .out)` where
    `.in`, `.out`, `.i`, and `.is_output` are available

  - **Function**: `function(i, in_dim, out_dim, is_output)` with
    signature as before

  The formula/function should return a named list of arguments to pass
  to the layer constructor. Available variables in formula context:

  - `.i` or `i`: Integer, the layer index (1-based)

  - `.in` or `in_dim`: Integer, input dimension for this layer

  - `.out` or `out_dim`: Integer, output dimension for this layer

  - `.is_output` or `is_output`: Logical, whether this is the final
    output layer

  If `NULL`, defaults to FFNN-style arguments:
  `list(in_dim, out_dim, bias = bias)`.

- forward_extract:

  Optional formula or function that processes layer outputs in the
  forward pass. Useful for layers that return complex structures (e.g.,
  RNNs return `list(output, hidden)`). Can be specified as:

  - **Formula**: `~ .[[1]]` or `~ .$output` where `.` represents the
    layer output

  - **Function**: `function(expr)` that accepts/returns a language
    object

  Common patterns:

  - Extract first element: `~ .[[1]]`

  - Extract named element: `~ .$output`

  - Extract with method: `~ .$get_output()`

  If `NULL`, layer outputs are used directly.

- before_output_transform:

  Optional formula or function that transforms input before the output
  layer. This is applied after the last hidden layer (and its
  activation) but before the output layer. Can be specified as:

  - **Formula**: `~ .[, .$size(2), ]` where `.` represents the current
    tensor

  - **Function**: `function(expr)` that accepts/returns a language
    object

  Common patterns:

  - Extract last timestep: `~ .[, .$size(2), ]`

  - Flatten: `~ .$flatten(start_dim = 1)`

  - Global pooling: `~ .$mean(dim = 2)`

  - Extract token: `~ .[, 1, ]`

  If `NULL`, no transformation is applied.

- after_output_transform:

  Optional formula or function that transforms the output after the
  output layer. This is applied after `self$out(x)` (the final layer)
  but before returning the result. Can be specified as:

  - **Formula**: `~ .$mean(dim = 2)` where `.` represents the output
    tensor

  - **Function**: `function(expr)` that accepts/returns a language
    object

  Common patterns:

  - Global average pooling: `~ .$mean(dim = 2)`

  - Squeeze dimensions: `~ .$squeeze()`

  - Reshape output: `~ .$view(c(-1, 10))`

  - Extract specific outputs: `~ .[, , 1:5]`

  If `NULL`, no transformation is applied.

- last_layer_args:

  Optional named list or formula specifying additional arguments for the
  output layer only. These arguments are appended to the output layer
  constructor after the arguments from `layer_arg_fn`. Can be specified
  as:

  - **Formula**: `~ list(kernel_size = 2L, bias = FALSE)`

  - **Named list**: `list(kernel_size = 2L, bias = FALSE)`

  This is useful when you need to override or add specific parameters to
  the final layer without affecting hidden layers. For example, in CNNs
  you might want a different kernel size for the output layer, or in
  RNNs you might want to disable bias in the final linear projection.
  Arguments in `last_layer_args` will override any conflicting arguments
  from `layer_arg_fn` when `.is_output = TRUE`. Default is an empty
  list.

- hd_neurons:

  Integer vector specifying the number of neurons (hidden units) in each
  hidden layer. The length determines the number of hidden layers in the
  network. Must contain at least one element.

- no_x:

  Integer specifying the number of input features (input dimension).

- no_y:

  Integer specifying the number of output features (output dimension).

- activations:

  Activation function specifications for hidden layers. Can be:

  - `NULL`: No activation functions applied

  - Character vector: e.g., `c("relu", "sigmoid", "tanh")`

  - `activation_spec` object: Created using
    [`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md),
    which allows specifying custom arguments. See examples.

  If a single activation is provided, it will be replicated across all
  hidden layers. Otherwise, the length should match the number of hidden
  layers.

- output_activation:

  Optional activation function for the output layer. Same format as
  `activations`, but should specify only a single activation. Common
  choices include `"softmax"` for classification or `"sigmoid"` for
  binary outcomes. Default is `NULL` (no output activation).

- bias:

  Logical indicating whether to include bias terms in layers. Default is
  `TRUE`. Note that this is passed to `layer_arg_fn` if provided, so
  custom layer argument functions should handle this parameter
  appropriately.

- eval:

  Logical indicating whether to evaluate the generated expression
  immediately. If `TRUE`, returns an instantiated `nn_module` class that
  can be called directly (e.g., `model()`). If `FALSE` (default),
  returns the unevaluated language expression that can be inspected or
  evaluated later with [`eval()`](https://rdrr.io/r/base/eval.html).
  Default is `FALSE`.

- .env:

  Default is [`parent.frame()`](https://rdrr.io/r/base/sys.parent.html).
  The environment in which the generated expression is to be evaluated

- ...:

  Additional arguments passed to layer constructors or for future
  extensions.

## Value

If `eval = FALSE` (default): A language object (unevaluated expression)
representing a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
definition. This expression can be evaluated with
[`eval()`](https://rdrr.io/r/base/eval.html) to create the module class,
which can then be instantiated with `eval(result)()` to create a model
instance.

If `eval = TRUE`: An instantiated `nn_module` class constructor that can
be called directly to create model instances (e.g., `result()`).

## Examples
