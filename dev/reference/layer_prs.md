# Layer argument pronouns for formula-based specifications

These pronouns provide a cleaner, more readable way to reference layer
parameters in formula-based specifications for
[`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)
and related functions. They work similarly to
[`rlang::.data`](https://rlang.r-lib.org/reference/dot-data.html) and
[`rlang::.env`](https://rlang.r-lib.org/reference/dot-data.html).

## Usage

``` r
.layer

.i

.in

.out

.is_output
```

## Format

Each pronoun is a zero-length object carrying only its class, which
`layer_arg_fn` formulas resolve to the corresponding per-layer value
while the module expression is being built. They hold no data of their
own and are never called directly.

## Details

Available pronouns:

- `.layer`: Access all layer parameters as a list-like object

- `.i`: Layer index (1-based integer)

- `.in`: Input dimension for the layer

- `.out`: Output dimension for the layer

- `.is_output`: Logical indicating if this is the output layer

These pronouns can be used in formulas passed to:

- `layer_arg_fn` parameter

- Custom layer configuration functions

## Usage

    # Using individual pronouns
    layer_arg_fn = ~ list(
        input_size = .in,
        hidden_size = .out,
        num_layers = if (.i == 1) 2L else 1L
    )

    # Using .layer pronoun (alternative syntax)
    layer_arg_fn = ~ list(
        input_size = .layer$ind,
        hidden_size = .layer$out,
        is_first = .layer$i == 1
    )

## Examples

``` r
# Pronouns stand in for per-layer values inside `layer_arg_fn` formulas.
# `.is_output` distinguishes the final layer from the hidden ones:
gru_arch = nn_arch(
    nn_name = "GRUNet",
    nn_layer = "torch::nn_gru",
    layer_arg_fn = ~ if (.is_output) {
        list(.in, .out)
    } else {
        list(input_size = .in, hidden_size = .out, batch_first = TRUE)
    }
)
gru_arch
#> 
#> ── Neural Network Architecture Spec 
#> • Name: GRUNet
#> • Layer: torch::nn_gru
#> • Out layer: same as nn_layer
#> • Input transform: none

# `.i` is the 1-based layer index; here only the first layer gets a bias.
nn_module_generator(
    nn_name = "TaperedNet",
    hd_neurons = c(16, 8),
    no_x = 4,
    no_y = 1,
    layer_arg_fn = ~ list(.in, .out, bias = .i == 1)
)
#> <quosure>
#> expr: ^torch::nn_module("TaperedNet", initialize = <function() {
#>           self$`torch::nn_linear_1` = torch::nn_linear(4, 16, bias = TRUE)
#>           self$`torch::nn_linear_2` = torch::nn_linear(16, 8, bias = FALSE)
#>           self$out = torch::nn_linear(8, 1, bias = FALSE)
#>         }>, forward = <function(x) {
#>           x = self$`torch::nn_linear_1`(x)
#>           x = self$`torch::nn_linear_2`(x)
#>           x = self$out(x)
#>           x
#>         }>)
#> env:  0x5632840e3a98
```
