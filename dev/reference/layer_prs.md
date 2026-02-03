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

An object of class `layer_pr` (inherits from `list`) of length 0.

An object of class `layer_index_pr` (inherits from `layer_pr`, `list`)
of length 0.

An object of class `layer_input_pr` (inherits from `layer_pr`, `list`)
of length 0.

An object of class `layer_output_pr` (inherits from `layer_pr`, `list`)
of length 0.

An object of class `layer_is_output_pr` (inherits from `layer_pr`,
`list`) of length 0.

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
