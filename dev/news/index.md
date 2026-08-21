# Changelog

## kindling (development version)

### New features

- [`save_kindling()`](https://kindling.joshuamarie.com/dev/reference/kindling-save-load.md)
  and
  [`load_kindling()`](https://kindling.joshuamarie.com/dev/reference/kindling-save-load.md)
  save and reload a trained model. Base
  [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) does not work for
  [kindling](https://kindling.joshuamarie.com) fits, since the
  underlying
  [`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
  holds external pointers that don’t survive plain R serialization.

- [`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
  gains `initial_model` to warm-start training from a previously trained
  (or reloaded) fit, and `track_optim_path` to optionally retain
  per-epoch loss, gradient norm, and a parameter hash in the returned
  object’s `$optim_path`.

- Training functions
  ([`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md),
  [`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md),
  [`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md))
  and [`predict()`](https://rdrr.io/r/stats/predict.html) now validate
  their inputs up front and error informatively on empty data,
  non-numeric predictors, missing/non-finite values, and length
  mismatches, instead of failing deep inside
  [torch](https://torch.mlverse.org/docs).

### Documentation

- Added a Life Cycle Statement to `CONTRIBUTING.md`.

- Package names in `Description` are consistently single-quoted.

- [`?train_nn`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
  gains sections clarifying training/validation/test data semantics, the
  missing-value policy (with a
  [recipes](https://github.com/tidymodels/recipes) imputation example),
  and guidance on learning rate, batch size, and epochs.

- [`vignette("tuning-capabilities")`](https://kindling.joshuamarie.com/dev/articles/tuning-capabilities.md)
  now points to custom
  [yardstick](https://github.com/tidymodels/yardstick) metrics via
  `metric_set()`.

### Ongoing features

- More visualization supports

  - NN architecture plot

### Fixes

- Lifting external package deps from all
  [tidymodels](https://tidymodels.tidymodels.org) interface, including
  the dependency to [tidyr](https://tidyr.tidyverse.org) and
  [dplyr](https://dplyr.tidyverse.org).

  - Functions relying on this packages is checked via
    [`rlang::check_installed()`](https://rlang.r-lib.org/reference/is_installed.html).

- Rewriting
  [`glue::glue()`](https://glue.tidyverse.org/reference/glue.html)
  dependency to [`paste0()`](https://rdrr.io/r/base/paste.html) /
  [`sprintf()`](https://rdrr.io/r/base/sprintf.html) for string
  interpolation.

### Bug Fixes

- `activations` specified as a
  [`list()`](https://rdrr.io/r/base/list.html) mixing named and unnamed
  elements (the documented syntax,
  e.g. `list(relu, tanh, softmax = args(dim = 2L))`) no longer crashes
  `parse_activation_spec()` with “missing value where TRUE/FALSE
  needed”.

  - The internal
    [`purrr::imap()`](https://purrr.tidyverse.org/reference/imap.html)
    call passed the element’s *name* as the position argument whenever
    the list had any names at all, instead of its integer index, so
    looking up `names(activations)[i]` silently returned `NA`.

  - Element names are now resolved once, up front, and iterated
    alongside each element via
    [`purrr::pmap()`](https://purrr.tidyverse.org/reference/pmap.html)
    instead of relying on `imap()`’s name-or-index behavior.

- Wrap `!requireNamespace(pkg, quietly = TRUE)` as this causes hidden
  bugs to `has_namespace()`

- `layer_prs` (the `.layer`, `.i`, `.in`, `.out`, and `.is_output`
  pronouns) is documented as a data topic again, with a `@format`
  description and runnable examples. A roxygen2 8.1.0 re-render had
  dropped its `\docType{data}` tag, which made documentation checks
  treat the pronouns as undocumented functions.

- `train_nn(verbose = TRUE)` now reports validation loss alongside
  training loss when `validation_split > 0`, matching
  [`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
  and
  [`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md).
  The validation branch computed a message but discarded it (a leftover
  from the
  [`glue::glue()`](https://glue.tidyverse.org/reference/glue.html) to
  [`sprintf()`](https://rdrr.io/r/base/sprintf.html) rewrite), so the
  validation loss never reached the console.

- [`table_summary()`](https://kindling.joshuamarie.com/dev/reference/table_summary.md)
  no longer iterates over `1:max(...)`, which produced a descending
  sequence when the table was empty.

- [`autoplot_diagnostics()`](https://kindling.joshuamarie.com/dev/reference/autoplot_diagnostics.md)
  errored on multi-output regression models instead of returning one
  actual-vs-fitted panel per output column. The length check comparing
  `actual` against the fitted values used
  [`length()`](https://rdrr.io/r/base/length.html) on a matrix (always
  2, the element count) instead of
  [`nrow()`](https://rdrr.io/r/base/nrow.html) (the observation count),
  so the mismatch check always failed before reaching the multi-output
  branch.

## kindling 0.3.2

CRAN release: 2026-07-10

### Bug fixes

- `"linear"` used as an `activations`/`output_activation` value no
  longer crashes at training time. It was previously resolved to
  [`torch::nnf_linear()`](https://torch.mlverse.org/docs/reference/nnf_linear.html)
  (an affine transform requiring its own weight/bias), instead of
  behaving as an identity/no-op activation. `"linear"` now consistently
  maps to [`identity()`](https://rdrr.io/r/base/identity.html)
  ([\#21](https://github.com/joshuamarie/kindling/issues/21)).

### Documentation

- Added missing `@examples` for
  [`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md),
  [`args()`](https://kindling.joshuamarie.com/dev/reference/args.md),
  and
  [`early_stop()`](https://kindling.joshuamarie.com/dev/reference/early_stop.md).

- Fixed the [tidymodels](https://tidymodels.tidymodels.org) example:
  loading `Ionosphere` via `box::use(mlbench[Ionosphere])` failed
  because `{mlbench}` does not export its datasets through
  `NAMESPACE`/`LazyData`. The example now uses
  `data(Ionosphere, package = "mlbench")` instead.

- Usage examples from `README` gets transferred to
  `vignettes/kindling.Rmd`.

## kindling 0.3.1

CRAN release: 2026-07-02

### New features

- [`autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html)
  and [`plot()`](https://rdrr.io/r/graphics/plot.default.html) methods
  for `nn_fit` objects now visualize the training loss history, with
  optional validation loss and an early-stopping marker when early
  stopping fires.

- [`autoplot_diagnostics()`](https://kindling.joshuamarie.com/dev/reference/autoplot_diagnostics.md)
  and
  [`plot_diagnostics()`](https://kindling.joshuamarie.com/dev/reference/autoplot_diagnostics.md)
  produce prediction diagnostic plots for `nn_fit` objects: residuals vs
  fitted and actual vs fitted panels for regression, one panel per
  output for multi-output regression, and a confusion matrix heatmap for
  classification.

- [vip](https://github.com/koalaverse/vip/) is removed from
  [kindling](https://kindling.joshuamarie.com)’s package dependencies,
  and moved to “Suggests” instead to avoid dependency redundancy.

## kindling 0.3.0

CRAN release: 2026-03-03

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

- [`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
  to execute
  [`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)

  - [`nn_arch()`](https://kindling.joshuamarie.com/dev/reference/nn_arch.md)
    must be supplied to inherit extra arguments from
    [`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)
    function.
  - Allows early stopping if `early_stopping` is supplied with
    [`early_stop()`](https://kindling.joshuamarie.com/dev/reference/early_stop.md).
  - Supported with several data types: `matrix`, `data.frame`, `dataset`
    ([torch](https://torch.mlverse.org/docs) dataset), and a formula
    interface.
  - [`train_nnsnip()`](https://kindling.joshuamarie.com/dev/reference/train_nnsnip.md)
    is now provided to bridge
    [`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
    with [tidymodels](https://tidymodels.tidymodels.org)

- You can supply customized activation function under
  [`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md)
  with
  [`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md).

  - Activation functions that especially don’t exist on
    `torch::nnf_*()`.
  - Supply the argument with a function
  - The function supplied into
    [`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md)
    must return a `torch` tensor object.
  - Example: `act_funs(new_act_fn(torch::torch_tanh))` or
    `act_funs(new_act_fn(\(x) torch::torch_tanh(x)))`
  - Use `.name` as a displayed name of the custom activation function.

#### Superset

- [`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md)
  as a DSL function now supports index-style parameter specification for
  parametric activation functions

  - Activation functions can now be modified using `[` syntax
    (e.g. `softplus[beta = 0.2]`)
  - The current
    [`args()`](https://kindling.joshuamarie.com/dev/reference/args.md)
    (e.g. `softplus = args(beta = 0.2)`) is now superseded by that.

#### Bug Fixes

- No suffix generated for `13` by
  [`ordinal_gen()`](https://kindling.joshuamarie.com/dev/reference/ordinal_gen.md).
  Now fixed.

- `hd_neurons` for both
  [`ffnn_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_gens.md)
  and
  [`rnn_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_gens.md)
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

- [`n_hlayers()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md)
  now fully supports tuning the number of hidden layers

- [`hidden_neurons()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md)
  gains support for discrete values via the `disc_values` argument

  - e.g. `disc_values = c(32L, 64L, 128L, 256L)`) is now allowed
  - This allows tuning over specific common hidden unit sizes instead of
    (or in addition to) a continuous range

### Implementation fixes

- Tuning methods and
  [`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md)
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

- The package description has been clarified

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
  [`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md)
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
