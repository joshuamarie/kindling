# kindling (development version)

## New features

-   Added regularization support for neural network models

    -   L1 regularization (Lasso) for feature selection via `mixture = 1`
    -   L2 regularization (Ridge) for weight decay via `mixture = 0`
    -   Elastic Net combining L1 and L2 penalties via `0 < mixture < 1`
    -   Controlled via `penalty` (regularization strength) and `mixture` (L1/L2 balance) parameters
    -   Follows tidymodels conventions for consistency with `glmnet` and other packages

-   `n_hlayers()` now fully supports tuning the number of hidden layers

-   `hidden_neurons()` gains support for discrete values via the `disc_values` argument 

    -   e.g. `disc_values = c(32L, 64L, 128L, 256L)`) is now allowed
    -   This allows tuning over specific common hidden unit sizes instead of (or in addition to) a continuous range

## Implementation fixes

-   Tuning methods and `grid_depth()` is now fixed

    -   Parameter space for the number of hidden layers is now fixed and active
    -   Corrected parameter space handling for `n_hlayers` (no more invalid sampling when `x > 1`)
    -   Uses `tidyr::expand_grid()`, not `purrr::cross*()`
    -   Fix randomization of parameter space which will produce NAs outside from `{kindling}`'s own 'dials'
    -   No more list columns when `n_hlayers = 1`

-   The supported models now use `hardhat::mold()`, instead of `model.frame()` and `model.matrix()`.

## Documentation

-   Add a vignette to showcase the comparison with other similar packages
-   The package description got few clarifications

# kindling 0.1.0

-   Initial CRAN release
-   Higher-level interface for torch package to define, train, and tune neural networks
-   Support for feedforward (multi-layer perceptron) and recurrent networks (RNN, LSTM, GRU)
-   Integration with tidymodels ecosystem (parsnip, workflows, recipes, tuning)
-   Variable importance plots and network visualization tools
