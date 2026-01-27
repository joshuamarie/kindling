# Changelog

## kindling (development version)

### New features

- Added regularization support for neural network models

  - L1 regularization (Lasso) for feature selection via `mixture = 1`
  - L2 regularization (Ridge) for weight decay via `mixture = 0`
  - Elastic Net combining L1 and L2 penalties via `0 < mixture < 1`
  - Controlled via `penalty` (regularization strength) and `mixture`
    (L1/L2 balance) parameters
  - Follows tidymodels conventions for consistency with `glmnet` and
    other packages

- Tuning methods and
  [`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md)
  is now fixed

  - Parameter space for the number of hidden layers is now fixed and
    active
  - Initial implementation uses
    [`sample()`](https://rdrr.io/r/base/sample.html) creates bug when
    `x` \> 1 for `type` != “regular”
  - Uses
    [`tidyr::expand_grid()`](https://tidyr.tidyverse.org/reference/expand_grid.html),
    not `purrr::cross*()`
  - Fix randomization of parameter space which will produce NAs outside
    from [kindling](https://kindling.joshuamarie.com)‘s own ’dials’
  - No more list columns when `n_hlayers = 1`

- Vignette to showcase the comparison with other similar packages

- `hidden_neurons` parameter now supports discrete values specification

  - Users can specify exact neuron counts via `values` parameter (e.g.,
    `hidden_neurons(values = c(32, 64, 128))`)
  - Maintains backward compatibility with range-based parameters (e.g.,
    `hidden_neurons(range = c(8L, 512L))` /
    `hidden_neurons(c(8L, 512L))`)

### Documentation improvements

- Added `\value` documentation to `kindling-nn-wrappers` for CRAN
  compliance
- Documented argument handling and list-column unwrapping in tidymodels
  wrapper functions
- Clarified the relationship between
  [`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md)
  and wrapper functions

## kindling 0.1.0

- Initial CRAN release
- Higher-level interface for torch package to define, train, and tune
  neural networks
- Support for feedforward (multi-layer perceptron) and recurrent
  networks (RNN, LSTM, GRU)
- Integration with tidymodels ecosystem (parsnip, workflows, recipes,
  tuning)
- Variable importance plots and network visualization tools
