# Changelog

## kindling (development version)

### New Experimental functions

- Generalized `nn_module()` expression generator to generate
  [`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
  expression for the same sequential NN architectures

  - This is how you use to generate `nn_module()` for 1D-CNN
    (Convolutional Neural Networks) with 3 hidden layers:

  ``` r
  nn_module_generator(
      nn_name = "CNN1DClassifier",
      nn_layer = "nn_conv1d",
      layer_arg_fn = ~ if (.is_output) {
          list(.in, .out)
      } else {
          list(
              in_channels = .in,
              out_channels = .out,
              kernel_size = 3L,
              stride = 1L,
              padding = 1L 
          )
      },
      after_output_transform = ~ .$mean(dim = 2),
      last_layer_args = list(kernel_size = 1, stride = 2),
      hd_neurons = c(16, 32, 64),
      no_x = 1,
      no_y = 10,
      activations = "relu"
  )
  ```

- [`train_nn()`](https://kindling.joshuamarie.com/reference/gen-nn-train.md)
  to execute
  [`nn_module_generator()`](https://kindling.joshuamarie.com/reference/nn_module_generator.md)

  - [`nn_arch()`](https://kindling.joshuamarie.com/reference/nn_arch.md)
    must be supplied to inherit extra arguments from
    [`nn_module_generator()`](https://kindling.joshuamarie.com/reference/nn_module_generator.md)
    function.
  - Allows early stopping if `early_stopping` is supplied with
    [`early_stop()`](https://kindling.joshuamarie.com/reference/early_stop.md).
  - [`train_nnsnip()`](https://kindling.joshuamarie.com/reference/train_nnsnip.md)
    is now provided to make
    [`train_nn()`](https://kindling.joshuamarie.com/reference/gen-nn-train.md)
    interfaced with [tidymodels](https://tidymodels.tidymodels.org)

#### Superset

- [`act_funs()`](https://kindling.joshuamarie.com/reference/act_funs.md)
  as a DSL function now supports index-style parameter specification for
  parametric activation functions

  - Activation functions can now be modified using `[` syntax
    (e.g. `softplus[beta = 0.2]`)
  - The current
    [`args()`](https://kindling.joshuamarie.com/reference/args.md)
    (e.g. `softplus = args(beta = 0.2)`) is now superseded by that.

#### Bug Fixes

- No suffix generated for `13` by
  [`ordinal_gen()`](https://kindling.joshuamarie.com/reference/ordinal_gen.md).
  Now fixed.

## kindling 0.2.1

### Fixes

- `hd_neurons` for both
  [`ffnn_generator()`](https://kindling.joshuamarie.com/reference/nn_gens.md)
  and
  [`rnn_generator()`](https://kindling.joshuamarie.com/reference/nn_gens.md)
  accepts empty arguments, which implies there’s no hidden layers
  applied.

## kindling 0.2.0

CRAN release: 2026-02-04

### New features

- Added regularization support for neural network models

  - L1 regularization (Lasso) for feature selection via `mixture = 1`
  - L2 regularization (Ridge) for weight decay via `mixture = 0`
  - Elastic Net combining L1 and L2 penalties via `0 < mixture < 1`
  - Controlled via `penalty` (regularization strength) and `mixture`
    (L1/L2 balance) parameters
  - Follows tidymodels conventions for consistency with `glmnet` and
    other packages

- [`n_hlayers()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  now fully supports tuning the number of hidden layers

- [`hidden_neurons()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  gains support for discrete values via the `disc_values` argument

  - e.g. `disc_values = c(32L, 64L, 128L, 256L)`) is now allowed
  - This allows tuning over specific common hidden unit sizes instead of
    (or in addition to) a continuous range

### Implementation fixes

- Tuning methods and
  [`grid_depth()`](https://kindling.joshuamarie.com/reference/grid_depth.md)
  is now fixed

  - Parameter space for the number of hidden layers is now fixed and
    active
  - Corrected parameter space handling for `n_hlayers` (no more invalid
    sampling when `x > 1`)
  - Uses
    [`tidyr::expand_grid()`](https://tidyr.tidyverse.org/reference/expand_grid.html),
    not `purrr::cross*()`
  - Fix randomization of parameter space which will produce NAs outside
    from [kindling](https://kindling.joshuamarie.com)‘s own ’dials’
  - No more list columns when `n_hlayers = 1`

- The supported models now use
  [`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html),
  instead of [`model.frame()`](https://rdrr.io/r/stats/model.frame.html)
  and [`model.matrix()`](https://rdrr.io/r/stats/model.matrix.html).

### Documentation

- Add a vignette to showcase the comparison with other similar packages

- The package description got few clarifications

- Vignette to showcase the comparison with other similar packages

- `hidden_neurons` parameter now supports discrete values specification

  - Users can specify exact neuron counts via `values` parameter (e.g.,
    `hidden_neurons(values = c(32, 64, 128))`)
  - Maintains backward compatibility with range-based parameters (e.g.,
    `hidden_neurons(range = c(8L, 512L))` /
    `hidden_neurons(c(8L, 512L))`)

- Added `\value` documentation to `kindling-nn-wrappers` for CRAN
  compliance

- Documented argument handling and list-column unwrapping in tidymodels
  wrapper functions

- Clarified the relationship between
  [`grid_depth()`](https://kindling.joshuamarie.com/reference/grid_depth.md)
  and wrapper functions

## kindling 0.1.0

CRAN release: 2026-01-31

- Initial CRAN release
- Higher-level interface for torch package to define, train, and tune
  neural networks
- Support for feedforward (multi-layer perceptron) and recurrent
  networks (RNN, LSTM, GRU)
- Integration with tidymodels ecosystem (parsnip, workflows, recipes,
  tuning)
- Variable importance plots and network visualization tools
