
<!-- README.md is generated from README.Rmd. Please edit that file -->

# kindling

<!-- badges: start -->

<!-- badges: end -->

> 🚧 **Under Construction**: This package is currently in early
> development. 🚧

`{kindling}` provides a higher-level interface that bridges the
`{torch}` and `{tidymodels}` ecosystems, making it easier to define,
train, and tune deep learning models in R. The package allows users to
work with multiple types of neural network architectures including:

- Feedforward neural networks (FNNs)
- Convolutional neural networks (CNNs)
- Recurrent neural networks (RNNs)

The package is designed to integrate seamlessly into `{tidymodels}`
workflows, bringing the power of deep learning to the familiar
tidymodels interface. It supports hyperparameter tuning for network
architecture components such as the number of hidden layers, units per
layer, and activation functions.

## Installation

You can install the development version of `{kindling}` from
[GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("joshuamarie/kindling")
```

## Features

- **Tidymodels integration**: Use familiar `{parsnip}` syntax to specify
  and train neural network models
- **Architecture tuning**: Tune the structure of your networks alongside
  traditional hyperparameters
- **Multiple architectures**: Support for feedforward, convolutional,
  and recurrent neural networks
- **Torch backend**: Leverages the `{torch}` package for efficient
  computation

## Getting Started

Documentation and examples are currently in development. Check back soon
for usage examples and vignettes.

## Dependencies

`{kindling}` builds on several key packages:

- `{torch}`: For neural network computation
- `{parsnip}`: For model specification
- `{tidymodels}`: For the broader modeling workflow

## License

MIT + file LICENSE
