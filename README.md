
<!-- README.md is generated from README.Rmd. Please edit that file -->

# kindling: Higher-level interface of torch package to auto-train neural networks <img src="man/figures/logo.png" align="right" alt="" width="120"/>

> 🚧 **Under Construction**: This package is currently in early
> development. 🚧

<!-- <!-- badges: start -->

–\>
<!-- [![R-CMD-check](https://github.com/joshuamarie/kindling/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/joshuamarie/kindling/actions/workflows/R-CMD-check.yaml) -->
<!-- <!-- badges: end --> –\>

## Overview

`{kindling}` bridges the gap between **{torch}** and **{tidymodels}**,
offering a streamlined interface for building, training, and tuning deep
learning models within the familiar `tidymodels` ecosystem.

Whether you’re prototyping neural architectures or deploying production
models, `{kindling}` minimizes boilerplate code while preserving the
flexibility of `{torch}`. It works seamlessly with `{parsnip}`,
`{recipes}`, and `{workflows}` to bring deep learning into your existing
modeling pipeline.

### Key Features

- Seamless integration with `parsnip` through `set_engine("kindling")`
- Native support for `{tidymodels}` workflows and pipelines
- Multiple architectures available: feedforward networks (DNN/FFNN) and
  recurrent variants (RNN, LSTM, GRU)
- Fine-grained control over network depth, layer sizes, and activation
  functions
- Full GPU acceleration via `{torch}` tensors
- Dramatically less boilerplate than raw `{torch}` implementations

### Supported Architectures

- **Feedforward Networks (DNN/FFNN)**: Classic multi-layer perceptrons
  for tabular data and general supervised learning
- **Recurrent Neural Networks (RNN)**: Basic recurrent architecture for
  sequential patterns
- **Long Short-Term Memory (LSTM)**: Sophisticated recurrent networks
  with gating mechanisms for long-range dependencies
- **Gated Recurrent Units (GRU)**: Streamlined alternative to LSTM with
  fewer parameters

## Installation

This package isn’t on CRAN yet. Install the development version from
GitHub:

``` r
# install.packages("pak")
pak::pak("joshuamarie/kindling")
```

## Usage: Four Levels of Interaction

`{kindling}` leverages R’s metaprogramming capabilities through *code
generation*. Generated `torch::nn_module` expressions power the training
functions, which in turn serve as engines for `{tidymodels}`
integration. This architecture gives you flexibility to work at whatever
abstraction level suits your task.

``` r
library(kindling)
#> 
#> Attaching package: 'kindling'
#> The following object is masked from 'package:base':
#> 
#>     args
```

### Level 1: Code Generation for `torch::nn_module`

At the lowest level, you can generate raw `torch::nn_module` code for
maximum customization. Functions ending with `_generator` return
unevaluated expressions you can inspect, modify, or execute.

Here’s how to generate a feedforward network specification:

``` r
ffnn_generator(
    nn_name = "MyFFNN",
    hd_neurons = c(64, 32, 16),
    no_x = 10,
    no_y = 1,
    activations = 'relu'
)
#> torch::nn_module("MyFFNN", initialize = function () 
#> {
#>     self$fc1 = torch::nn_linear(10, 64, bias = TRUE)
#>     self$fc2 = torch::nn_linear(64, 32, bias = TRUE)
#>     self$fc3 = torch::nn_linear(32, 16, bias = TRUE)
#>     self$out = torch::nn_linear(16, 1, bias = TRUE)
#> }, forward = function (x) 
#> {
#>     x = self$fc1(x)
#>     x = torch::nnf_relu(x)
#>     x = self$fc2(x)
#>     x = torch::nnf_relu(x)
#>     x = self$fc3(x)
#>     x = torch::nnf_relu(x)
#>     x = self$out(x)
#>     x
#> })
```

This creates a three-hidden-layer network (64 - 32 - 16 neurons) that
takes 10 inputs and produces 1 output. Each hidden layer uses ReLU
activation, while the output layer remains “untransformed”.

### Level 2: Direct Training Interface

Skip the code generation and train models directly with your data. This
approach handles all the `{torch}` boilerplate internally.

Let’s classify iris species:

``` r
model = ffnn(
    Species ~ .,
    data = iris,
    hidden_neurons = c(64, 32),
    activations = act_funs(relu, softshrink = args(lambd = 0.5)),
    loss = "cross_entropy",
    epochs = 100
)
model
```


    ======================= Feedforward Neural Networks (MLP) ======================


    -- FFNN Model Summary ----------------------------------------------------------

          --------------------------------------------------------------------
            NN Model Type                        FFNN    n_predictors      4
            Number of Epochs                      100    n_response        3
            Hidden Layer Units                 64, 32                       
            Number of Hidden Layers                 2                       
            Pred. Type                 classification                       
          --------------------------------------------------------------------



    -- Activation function ---------------------------------------------------------

                    ------------------------------------------------
                      1st Layer {64}                          relu
                      2nd Layer {32}       softshrink(lambd = 0.5)
                      Output Activation    No act function applied
                    ------------------------------------------------

The `predict()` method offers flexible prediction behavior through its
`newdata` argument:

1.  **Without new data** — predictions default to the training set:

    ``` r
    predict(model) |> 
        (\(x) table(actual = iris$Species, predicted = x))()
    #>             predicted
    #> actual       setosa versicolor virginica
    #>   setosa         50          0         0
    #>   versicolor      0         47         3
    #>   virginica       0          0        50
    ```

2.  **With new data** — simply pass a data frame:

    ``` r
    sample_iris = dplyr::slice_sample(iris, n = 10, by = Species)

    predict(model, newdata = sample_iris) |> 
        (\(x) table(actual = sample_iris$Species, predicted = x))()
    #>             predicted
    #> actual       setosa versicolor virginica
    #>   setosa         10          0         0
    #>   versicolor      0         10         0
    #>   virginica       0          0        10
    ```

### Level 3: Full tidymodels Integration

Work with neural networks just like any other `{parsnip}` model. This
unlocks the entire `{tidymodels}` toolkit for preprocessing,
cross-validation, and model evaluation.

``` r
# library(kindling)
# library(parsnip)
# library(yardstick)
box::use(
    kindling[mlp_kindling, rnn_kindling, act_funs, args],
    parsnip[fit, augment],
    yardstick[metrics],
    mlbench[Ionosphere] # data(Ionosphere, package = "mlbench")
)

ionosphere_data = Ionosphere[, -2]

# Train a feedforward network with parsnip
mlp_kindling(
    mode = "classification",
    hidden_neurons = c(128, 64),
    activations = act_funs(relu, softshrink = args(lambd = 0.5)),
    epochs = 100
) |>
    fit(Class ~ ., data = ionosphere_data) |>
    augment(new_data = ionosphere_data) |>
    metrics(truth = Class, estimate = .pred_class)
#> # A tibble: 2 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.989
#> 2 kap      binary         0.975

# Or try a recurrent architecture (demonstrative example with tabular data)
rnn_kindling(
    mode = "classification",
    hidden_neurons = c(128, 64),
    activations = act_funs(relu, elu),
    epochs = 100,
    rnn_type = "gru"
) |>
    fit(Class ~ ., data = ionosphere_data) |>
    augment(new_data = ionosphere_data) |>
    metrics(truth = Class, estimate = .pred_class)
#> # A tibble: 2 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.641
#> 2 kap      binary         0
```

### Level 4: Hyperparameter Tuning & Resampling

> **Coming Soon**: This functionality is planned but not yet available.

The roadmap includes full support for hyperparameter tuning via `{tune}`
with searchable parameters:

- Network depth (number of hidden layers)
- Layer widths (neurons per layer)
- Activation function combinations

Resampling strategies from `{rsample}` will enable robust
cross-validation workflows, orchestrated through the `{tune}` and
`{dials}` APIs.

## References

Falbel D, Luraschi J (2023). *torch: Tensors and Neural Networks with
‘GPU’ Acceleration*. R package version 0.13.0,
<https://torch.mlverse.org>, <https://github.com/mlverse/torch>.

Wickham H (2019). *Advanced R*, 2nd edition. Chapman and Hall/CRC. ISBN
978-0815384571, <https://adv-r.hadley.nz/>.

Goodfellow I, Bengio Y, Courville A (2016). *Deep Learning*. MIT Press.
<https://www.deeplearningbook.org/>.

## License

MIT + file LICENSE

## Code of Conduct

Please note that the kindling project is released with a [Contributor
Code of
Conduct](https://contributor-covenant.org/version/2/1/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.
