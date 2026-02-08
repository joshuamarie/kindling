# `{kindling}`: Higher-level interface of torch package to auto-train neural networks

`{kindling}` enables R users to build and train deep neural networks
such as:

- Deep Neural Networks / (Deep) Feedforward Neural Networks (DNN / FFNN)

- Recurrent Neural Networks (RNN)

It is mainly designed to generate code expressions of the current
architectures which happens to reduce boilerplate `{torch}` code for the
said current architectures. It also integrate seamlessly with titanic ML
frameworks - currently with `{tidymodels}`, which enables components
like `{parsnip}`, `{recipes}`, and `{workflows}`, allowing ergonomic
interface for model specification, training, and evaluation.

Thus, the package supports hyperparameter tuning for:

- Number of hidden layers

- Number of units per layer

- Choice of activation functions

Clarification: The hyperparameter tuning support is supported in version
0.1.0, but `n_hlayer()` dial parameter is not supported. This changes
after 0.2.0 release.

## Details

The `{kindling}` package provides a unified, high-level interface that
bridges the `{torch}` and `{tidymodels}` ecosystems, making it easy to
define, train, and tune deep learning models using the familiar
`tidymodels` workflow.

## How to use

The following uses of this package has 3 levels:

Level 1: Code generation

    ffnn_generator(
        nn_name = "MyFFNN",
        hd_neurons = c(64, 32, 16),
        no_x = 10,
        no_y = 1,
        activations = 'relu'
    )

Level 2: Direct Execution

    ffnn(
        Species ~ .,
        data = iris,
        hidden_neurons = c(128, 64, 32),
        activations = 'relu',
        loss = "cross_entropy",
        epochs = 100
    )

Level 3: Conventional tidymodels interface

    # library(parsnip)
    # library(kindling)
    box::use(
       kindling[mlp_kindling, rnn_kindling, act_funs, args],
       parsnip[fit, augment],
       yardstick[metrics],
       mlbench[Ionosphere] # data(Ionosphere, package = "mlbench")
    )

    # Remove V2 as it's all zeros
    ionosphere_data = Ionosphere[, -2]

    # MLP example
    mlp_kindling(
        mode = "classification",
        hidden_neurons = c(128, 64),
        activations = act_funs(relu, softshrink = args(lambd = 0.5)),
        epochs = 100
    ) |>
        fit(Class ~ ., data = ionosphere_data) |>
        augment(new_data = ionosphere_data) |>
        metrics(truth = Class, estimate = .pred_class)
    #> A tibble: 2 × 3
    #>   .metric  .estimator .estimate
    #>   <chr>    <chr>          <dbl>
    #> 1 accuracy binary         0.989
    #> 2 kap      binary         0.975

    # RNN example (toy usage on non-sequential data)
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
    #> A tibble: 2 × 3
    #>   .metric  .estimator .estimate
    #>   <chr>    <chr>          <dbl>
    #> 1 accuracy binary         0.641
    #> 2 kap      binary         0

## Main Features

- Code generation of `{torch}` expression

- Multiple architectures available: feedforward networks (MLP/DNN/FFNN)
  and recurrent variants (RNN, LSTM, GRU)

- Native support for `{tidymodels}` workflows and pipelines

- Fine-grained control over network depth, layer sizes, and activation
  functions

- GPU acceleration supports via `{torch}` tensors

## License

MIT + file LICENSE

## References

Falbel D, Luraschi J (2023). *torch: Tensors and Neural Networks with
'GPU' Acceleration*. R package version 0.13.0,
<https://torch.mlverse.org>, <https://github.com/mlverse/torch>.

Wickham H (2019). *Advanced R*, 2nd edition. Chapman and Hall/CRC. ISBN
978-0815384571, <https://adv-r.hadley.nz/>.

Goodfellow I, Bengio Y, Courville A (2016). *Deep Learning*. MIT Press.
<https://www.deeplearningbook.org/>.

## See also

Useful links:

- <https://kindling.joshuamarie.com>

- <https://joshuamarie.github.io/kindling>

- <https://github.com/joshuamarie/kindling>

- Report bugs at <https://github.com/joshuamarie/kindling/issues>

## Author

**Maintainer**: Joshua Marie <joshua.marie.k@gmail.com>
