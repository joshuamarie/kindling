# kindling (development version)

## New Experimental functions

-   Generalized `nn_module()` expression generator to generate `torch::nn_module()` expression for the same sequential NN architectures

    -  This is how you use to generate `nn_module()` for 1D-CNN (Convolutional Neural Networks) with 3 hidden layers:
    
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

-   `train_nn()` to execute `nn_module_generator()`

    -   `nn_arch()` must be supplied to inherit extra arguments from `nn_module_generator()` function. 

### Superset

-  `act_funs()` as a DSL function now supports index-style parameter specification for parametric activation functions

    -   Activation functions can now be modified using `[` syntax (e.g. `softplus[beta = 0.2]`)
    -   The current `args()` (e.g. `softplus = args(beta = 0.2)`) is now superseded by that. 

# kindling 0.2.1

## Fixes

-  `hd_neurons` for both `ffnn_generator()` and `rnn_generator()` accepts empty arguments, which implies there's no hidden layers applied. 

# kindling 0.2.0

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
-   Vignette to showcase the comparison with other similar packages
-   `hidden_neurons` parameter now supports discrete values specification

    -   Users can specify exact neuron counts via `values` parameter (e.g., `hidden_neurons(values = c(32, 64, 128))`)
    -   Maintains backward compatibility with range-based parameters (e.g., `hidden_neurons(range = c(8L, 512L))` / `hidden_neurons(c(8L, 512L))`)

-   Added `\value` documentation to `kindling-nn-wrappers` for CRAN compliance
-   Documented argument handling and list-column unwrapping in tidymodels wrapper functions
-   Clarified the relationship between `grid_depth()` and wrapper functions

# kindling 0.1.0

-   Initial CRAN release
-   Higher-level interface for torch package to define, train, and tune neural networks
-   Support for feedforward (multi-layer perceptron) and recurrent networks (RNN, LSTM, GRU)
-   Integration with tidymodels ecosystem (parsnip, workflows, recipes, tuning)
-   Variable importance plots and network visualization tools
