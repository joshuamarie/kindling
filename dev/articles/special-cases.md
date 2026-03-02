# Special Cases: Linear and Logistic Regression

## What’s so special about {kindling}

This package is planned to make it compatible for any machine learning
task, even time series and image classification cam be supported. Yes,
you can do both linear regression and logistic regression with extra
steps: heavily customized optimizer and loss functions. The
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
function (available on \>v0.3.x) supports this { `optimizer`
\leftrightarrow `optimizer_args` } and { `loss` }. For both cases, the
key is to remove all hidden layers and rely entirely on the output layer
and the appropriate loss function to recover the classical model’s
behavior.

## Setup

``` r
box::use(
    kindling[train_nn, act_funs, args],
    recipes[
        recipe, step_dummy, step_normalize,
        all_nominal_predictors, all_numeric_predictors
    ],
    rsample[initial_split, training, testing],
    yardstick[metric_set, rmse, rsq, accuracy, mn_log_loss],
    dplyr[mutate, select],
    tibble[tibble]
)
```

## Linear Regression as a Special Case

A standard linear regression model predicts a continuous outcome as a
weighted sum of inputs — no nonlinearity, no hidden layers. A neural
network recovers this exactly when:

- There are *no hidden layers* (`hidden_neurons = integer(0)` or simply
  omit it),
- The *output activation is the identity* (i.e., no activation), and
- The common loss function is MSE, but we can choose different loss
  function: (`loss = "mse"`).

Under these conditions, gradient descent minimizes the same objective as
ordinary least squares, and the learned weights converge to the OLS
solution given sufficient epochs and a small learning rate.

### Data

We use `mtcars` to predict fuel efficiency (`mpg`) from the other
variables.

``` r
set.seed(42)
split = initial_split(mtcars, prop = 0.8)
train = training(split)
test = testing(split)

rec = recipe(mpg ~ ., data = train) |>
    step_normalize(all_numeric_predictors())
```

### Fitting the model

To create no hidden units, the `hidden_neuron` parameter from
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
considers the following to achieve:

1.  `NULL`
2.  Empty [`c()`](https://rdrr.io/r/base/c.html)
3.  No arguments at all

In this example, the empty vector [`c()`](https://rdrr.io/r/base/c.html)
is used and will collapse the network to a single linear layer from
inputs to output. The `optimizer = "rmsprop"` with a small `learn_rate`
mirrors classical gradient descent for OLS.

``` r
lm_nn = train_nn(
    mpg ~ .,
    data = train,
    hidden_neurons = c(),
    loss = torch::nnf_l1_loss,
    optimizer = "rmsprop", 
    learn_rate = 0.01,
    epochs = 200,
    verbose = FALSE
)

lm_nn
```

    ## 
    ## ========================== Generalized Neural Network ==========================
    ## 
    ## 
    ## -- Model Summary ---------------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------------------------
    ##   NN Model Type           :         FFNN    n_predictors :     10
    ##   Number of Epochs        :          200    n_response   :      1
    ##   Hidden Layer Units      :                 reg.         :   None
    ##   Number of Hidden Layers :            0    Device       :    cpu
    ##   Pred. Type              :   regression                 :       
    ## -------------------------------------------------------------------
    ## 
    ## 
    ## 
    ## -- Activation Functions --------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------
    ##    Layer {}         :   No act function applied
    ##   Output Activation :   No act function applied
    ## -------------------------------------------------
    ## 
    ## 
    ## 
    ## -- Architecture Spec -----------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## --------------------------------------------------------------
    ##   nn_layer        :   N/A    before_output_transform :   N/A
    ##   out_nn_layer    :   N/A    after_output_transform  :   N/A
    ##   nn_layer_args   :   N/A    last_layer_args         :   N/A
    ##   layer_arg_fn    :   N/A    input_transform         :   N/A
    ##   forward_extract :   N/A                            :      
    ## --------------------------------------------------------------

### Evaluation

``` r
preds = predict(lm_nn, newdata = test)

tibble(
    truth = test$mpg,
    estimate = preds
) |>
    metric_set(rmse, rsq)(truth = truth, estimate = estimate)
```

    ## # A tibble: 2 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       4.43 
    ## 2 rsq     standard       0.944

### Comparison with `lm()`

``` r
lm_fit = lm(mpg ~ ., data = train)

tibble(
    truth = test$mpg,
    estimate = predict(lm_fit, newdata = test)
) |>
    metric_set(rmse, rsq)(truth = truth, estimate = estimate)
```

    ## # A tibble: 2 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       4.88 
    ## 2 rsq     standard       0.493

The two models should produce very similar RMSE and R^2 values. Any
small gap reflects that gradient descent is an iterative approximation,
while [`lm()`](https://rdrr.io/r/stats/lm.html) solves for the exact OLS
coefficients directly. Increasing `epochs` or switching to
`optimizer = "lbfgs"` (if supported) will close the gap further.

## Logistic Regression as a Special Case

Logistic regression models a binary or multiclass outcome by passing a
linear combination of inputs through a sigmoid or softmax activation. A
neural network with:

- **No hidden layers**,
- A **sigmoid output** for binary classification (or softmax for
  multiclass), and
- **Cross-entropy** (`loss = "cross_entropy"`) for the loss function

is mathematically equivalent to logistic regression.

### Binary Logistic Regression

We use the `Sonar` dataset from `{mlbench}` to distinguish rocks from
mines (binary outcome).

``` r
data("Sonar", package = "mlbench")

sonar = Sonar
set.seed(42)
split_s = initial_split(sonar, prop = 0.8, strata = Class)
train_s = training(split_s)
test_s = testing(split_s)

rec_s = recipe(Class ~ ., data = train_s) |>
    step_normalize(all_numeric_predictors())
```

``` r
logit_nn = train_nn(
    Class ~ .,
    data = train_s,
    hidden_neurons = c(),
    loss = "cross_entropy",
    optimizer = "adam",
    learn_rate = 0.01,
    epochs = 200,
    verbose = FALSE
)

logit_nn
```

    ## 
    ## ========================== Generalized Neural Network ==========================
    ## 
    ## 
    ## -- Model Summary ---------------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -----------------------------------------------------------------------
    ##   NN Model Type           :             FFNN    n_predictors :     60
    ##   Number of Epochs        :              200    n_response   :      2
    ##   Hidden Layer Units      :                     reg.         :   None
    ##   Number of Hidden Layers :                0    Device       :    cpu
    ##   Pred. Type              :   classification                 :       
    ## -----------------------------------------------------------------------
    ## 
    ## 
    ## 
    ## -- Activation Functions --------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------
    ##    Layer {}         :   No act function applied
    ##   Output Activation :   No act function applied
    ## -------------------------------------------------
    ## 
    ## 
    ## 
    ## -- Architecture Spec -----------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## --------------------------------------------------------------
    ##   nn_layer        :   N/A    before_output_transform :   N/A
    ##   out_nn_layer    :   N/A    after_output_transform  :   N/A
    ##   nn_layer_args   :   N/A    last_layer_args         :   N/A
    ##   layer_arg_fn    :   N/A    input_transform         :   N/A
    ##   forward_extract :   N/A                            :      
    ## --------------------------------------------------------------

``` r
preds_s = predict(logit_nn, newdata = test_s, type = "response")

tibble(
    truth = test_s$Class,
    estimate = preds_s
) |>
    accuracy(truth = truth, estimate = estimate)
```

    ## # A tibble: 1 × 3
    ##   .metric  .estimator .estimate
    ##   <chr>    <chr>          <dbl>
    ## 1 accuracy binary         0.744

### Comparison with `glm()` / `nnet::multinom()`

``` r
box::use(nnet[multinom])

glm_fit = glm(Class ~ ., data = train_s, family = binomial())
```

    ## Warning: glm.fit: algorithm did not converge

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
tibble(
    truth = test_s$Class,
    estimate = {
        as.factor({
            preds = predict(glm_fit, newdata = test_s, type = "response")
            ifelse(preds < 0.5, "M", "R")
        })
    }
) |>
    accuracy(truth = truth, estimate = estimate)
```

    ## # A tibble: 1 × 3
    ##   .metric  .estimator .estimate
    ##   <chr>    <chr>          <dbl>
    ## 1 accuracy binary         0.698

Again, accuracy should be comparable between the two approaches. The
neural network version converges iteratively, so the match is not
guaranteed to be exact, but both are optimizing the same cross-entropy
objective over a linear model.
