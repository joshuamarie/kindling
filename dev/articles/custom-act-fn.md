# Custom Activation Function

## Rationale

The biggest strength of [kindling](https://kindling.joshuamarie.com)
when modelling neural networks is its versatility — it inherits
[torch](https://torch.mlverse.org/docs)’s versatility while being human
friendly, including the ability to apply custom optimizer functions,
loss functions, and per-layer activation functions. Learn more:
<https://kindling.joshuamarie.com/articles/special-cases>.

With
[`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md),
you are not limited to the activation functions available in
[torch](https://torch.mlverse.org/docs)’s namespace. Use
[`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md)
to wrap any compatible function into a validated custom activation. This
feature, however, only available on version 0.3.0 and above.

## Function to use

To do this, use
[`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md)
and it takes a user-supplied function, validates it against a small
dummy tensor at *definition time* (a dry-run probe), and wraps it in a
call-time type guard. This means errors surface early — before your
model ever starts training.

The function you supply must:

- Accept at least one argument (the input tensor).
- Return a `torch_tensor`.

### Basic Usage

Currently, `nnf_tanh` doesn’t exist in
[torch](https://torch.mlverse.org/docs) namespace, so `tanh` argument is
not valid. With
[`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md),
you can wrap
[`torch::torch_tanh()`](https://torch.mlverse.org/docs/reference/torch_tanh.html)
to make it usable.

Here’s a basic example that wraps
[`torch::torch_tanh()`](https://torch.mlverse.org/docs/reference/torch_tanh.html)
as a custom activation:

``` r
hyper_tan = new_act_fn(\(x) torch::torch_tanh(x))
```

You can also pass it directly into
[`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md),
just like any built-in activation:

``` r
act_funs(relu, elu, new_act_fn(\(x) torch::torch_tanh(x)))
```

### Using Custom Activations in a Model

Naturally, functions for modelling like
[`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
accepts
[`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md)
into the `activations` argument. Again, you can pass a custom activation
function within
[`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md),
then pass it through
[`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md).

Here’s a basic example:

``` r
model = ffnn(
    Sepal.Length ~ .,
    data = iris[, 1:4],
    hidden_neurons = c(64, 32, 16),
    activations = act_funs(
        relu,
        silu,
        new_act_fn(\(x) torch::torch_tanh(x))
    ),
    epochs = 50
)
model
```

    ## 
    ## ======================= Feedforward Neural Networks (MLP) ======================
    ## 
    ## 
    ## -- FFNN Model Summary ----------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------------------------
    ##   NN Model Type           :         FFNN    n_predictors :      3
    ##   Number of Epochs        :           50    n_response   :      1
    ##   Hidden Layer Units      :   64, 32, 16    reg.         :   None
    ##   Number of Hidden Layers :            3    Device       :    cpu
    ##   Pred. Type              :   regression                 :       
    ## -------------------------------------------------------------------
    ## 
    ## 
    ## 
    ## -- Activation function ---------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------
    ##   1st Layer {64}    :                      relu
    ##   2nd Layer {32}    :                      silu
    ##   3rd Layer {16}    :                  <custom>
    ##   Output Activation :   No act function applied
    ## -------------------------------------------------

Each element of
[`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md)
corresponds to one hidden layer, in order. Here, the first hidden layer
uses ReLU, the second uses SiLU (Swish), and the third uses Tanh.

You can also use a single custom activation recycled across all layers:

``` r
ffnn(
    Sepal.Length ~ .,
    data = iris[, 1:4],
    hidden_neurons = c(64, 32),
    activations = act_funs(new_act_fn(\(x) torch::torch_tanh(x))),
    epochs = 50
)
```

    ## 
    ## ======================= Feedforward Neural Networks (MLP) ======================
    ## 
    ## 
    ## -- FFNN Model Summary ----------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------------------------
    ##   NN Model Type           :         FFNN    n_predictors :      3
    ##   Number of Epochs        :           50    n_response   :      1
    ##   Hidden Layer Units      :       64, 32    reg.         :   None
    ##   Number of Hidden Layers :            2    Device       :    cpu
    ##   Pred. Type              :   regression                 :       
    ## -------------------------------------------------------------------
    ## 
    ## 
    ## 
    ## -- Activation function ---------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------
    ##   1st Layer {64}    :                  <custom>
    ##   2nd Layer {32}    :                  <custom>
    ##   Output Activation :   No act function applied
    ## -------------------------------------------------

### Skipping the Dry-Run Probe

By default,
[`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md)
runs a quick dry-run with a small dummy tensor to validate your function
before training. You can disable this with `probe = FALSE`, though this
is generally not recommended:

``` r
my_act = new_act_fn(\(x) torch::torch_tanh(x), probe = FALSE)
```

### Naming Your Custom Activation

You can provide a human-readable name via `.name`, which is used in
print output and diagnostics:

``` r
my_act = new_act_fn(\(x) torch::torch_tanh(x), .name = "my_tanh")
```

Here’s a simple application:

``` r
ffnn(
    Sepal.Length ~ .,
    data = iris[, 1:4],
    hidden_neurons = c(64, 32),
    activations = act_funs(
        relu, 
        new_act_fn(\(x) torch::torch_tanh(x), .name = "hyper_tanh")
    ),
    epochs = 50
)
```

    ## 
    ## ======================= Feedforward Neural Networks (MLP) ======================
    ## 
    ## 
    ## -- FFNN Model Summary ----------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------------------------
    ##   NN Model Type           :         FFNN    n_predictors :      3
    ##   Number of Epochs        :           50    n_response   :      1
    ##   Hidden Layer Units      :       64, 32    reg.         :   None
    ##   Number of Hidden Layers :            2    Device       :    cpu
    ##   Pred. Type              :   regression                 :       
    ## -------------------------------------------------------------------
    ## 
    ## 
    ## 
    ## -- Activation function ---------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------
    ##   1st Layer {64}    :                      relu
    ##   2nd Layer {32}    :                hyper_tanh
    ##   Output Activation :   No act function applied
    ## -------------------------------------------------

## Error Handling

[`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md)
is designed to fail loudly and early. Common errors include:

1.  Function returns a non-tensor. This will error at definition time:

    ``` r
    new_act_fn(\(x) as.numeric(x))
    ```

        ## Error in `.assert_tensor_output()`:
        ## ! Dry-run must be a <torch_tensor>.
        ## ✖ Got <numeric>.
        ## ℹ Ensure your function returns the result of a torch operation.

2.  Function accepts no arguments. This will error immediately:

    ``` r
    new_act_fn(function() torch::torch_zeros(2))
    ```

        ## Error in `new_act_fn()`:
        ## ! `fn` must accept at least one argument (the input tensor).
        ## ℹ Use a lambda like `\(x) torch::torch_tanh(x)`.

These checks ensure your model’s architecture is valid before any data
ever flows through it.

## Summary

| Feature                                                                                    | Details                                                  |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------|
| Wraps any R function                                                                       | Must accept a tensor, return a tensor                    |
| Dry-run probe                                                                              | Validates at definition time (`probe = TRUE` by default) |
| Call-time guard                                                                            | Type-checks output on every forward pass                 |
| Compatible with [`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md) | Use alongside built-in activations freely                |
| Closures supported                                                                         | Parametric activations work naturally                    |
