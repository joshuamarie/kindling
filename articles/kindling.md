# Getting Started with kindling

## Introduction

[kindling](https://packages.joshuamarie.com/kindling) bridges the gap
between [torch](https://torch.mlverse.org/docs) and
[tidymodels](https://tidymodels.tidymodels.org), providing a streamlined
interface for building, training, and tuning deep learning models. This
vignette will guide you through the basic usage.

## Installation

``` r
# Install from GitHub
pak::pak("joshuamarie/kindling")
```

``` r
library(kindling)
#> 
#> Attaching package: 'kindling'
#> The following object is masked from 'package:base':
#> 
#>     args
```

## Four Levels of Interaction

[kindling](https://packages.joshuamarie.com/kindling) offers flexibility
through four levels of abstraction:

1.  **Code Generation** - Generate raw
    [`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
    code
2.  **Direct Training** - Train models with simple function calls
3.  **tidymodels Integration** - Use with `parsnip`, `recipes`, and
    `workflows`
4.  **Hyperparameter Tuning** - Optimize models with `tune` and `dials`

## Level 1: Code Generation

Generate PyTorch-style module code:

``` r
ffnn_generator(
    nn_name = "MyNetwork",
    hd_neurons = c(64, 32),
    no_x = 10,
    no_y = 1,
    activations = 'relu'
)
```

## Level 2: Direct Training

Train a model with one function call:

``` r
model = ffnn(
    Species ~ .,
    data = iris,
    hidden_neurons = c(10, 15, 7),
    activations = act_funs(relu, elu),
    loss = "cross_entropy",
    epochs = 100
)

predictions = predict(model, newdata = iris)
```

## Level 3: tidymodels Integration

Work with neural networks like any other `parsnip` model:

``` r
box::use(
    parsnip[fit, augment],
    yardstick[metrics]
)

nn_spec = mlp_kindling(
    mode = "classification",
    hidden_neurons = c(10, 7),
    activations = act_funs(relu, softshrink = args(lambd = 0.5)),
    epochs = 100
)

nn_fit = fit(nn_spec, Species ~ ., data = iris)
augment(nn_fit, new_data = iris) |> 
    metrics(truth = Species, estimate = .pred_class)
```

## Learn More

- Read the [README](https://packages.joshuamarie.com/kindling/index.md)
  for comprehensive examples
- Browse the [function
  reference](https://packages.joshuamarie.com/kindling/reference/index.md)
- Visit the [blog](https://joshuamarie.com) for tutorials
