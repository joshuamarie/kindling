# Tuning Capabilities

## Rationale

How capable this package when tuning neural networks? One of the
package’s capabilities is the ability to fine-tune the whole
architecture, and this includes the depth of the architecture — not
limited to the number of hidden neurons, also includes the number of
layers. Neural networks with [torch](https://torch.mlverse.org/docs)
natively supports different activation functions for different layers,
thus [kindling](https://kindling.joshuamarie.com) supports:

- The **number of hidden layers** (depth)
- The **number of neurons per layer** (width)
- The **activation function per layer**, including parametric variants
  (e.g. `softshrink(lambd = 0.2)`)

## Custom grid creation

[kindling](https://kindling.joshuamarie.com) has its own function to
define the grid which includes the depth of the architecture:
[`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md),
an analogue function to
[`dials::grid_space_filling()`](https://dials.tidymodels.org/reference/grid_space_filling.html),
except it creates `"regular"` grid. You can tweak `n_hlayer` parameter,
and you can define the grid that has the depth. This parameter can be
scalar (e.g. `2`), integer vector (e.g. `1:2`), and/or using a
[dials](https://dials.tidymodels.org) function called `n_hlayer()`. When
`n_hlayer` is greater than 2, the certain parameters `hidden_neurons`
and `activations` creates a list-column, which contains vectors for each
parameter grid, depending on `n_hlayer` you defined.

## Setup

We won’t stop you from using
[`library()`](https://rdrr.io/r/base/library.html) function, but we
strongly recommend using
[`box::use()`](http://klmr.me/box/reference/use.md) and explicitly
import the names from the namespaces you want to attach.

``` r
# library(kindling)
# library(tidymodels)
# library(modeldata)

box::use(
    kindling[mlp_kindling, act_funs, args, hidden_neurons, activations, grid_depth],
    dplyr[select, ends_with, mutate, slice_sample],
    tidyr[drop_na],
    rsample[initial_split, training, testing, vfold_cv],
    recipes[
        recipe, step_dummy, step_normalize,
        all_nominal_predictors, all_numeric_predictors
    ],
    modeldata[penguins],
    parsnip[tune, set_mode, fit, augment],
    workflows[workflow, add_recipe, add_model],
    dials[learn_rate],
    tune[tune_grid, show_best, collect_metrics, select_best, finalize_workflow, last_fit],
    yardstick[metric_set, rmse, rsq],
    ggplot2[autoplot]
)
```

We’ll use the `penguins` dataset from
[modeldata](https://modeldata.tidymodels.org) to predict body mass (in
kilograms) from physical measurements — a straightforward regression
task that lets us focus on the tuning workflow.

## Usage

[kindling](https://kindling.joshuamarie.com) provides the
[`mlp_kindling()`](https://kindling.joshuamarie.com/dev/reference/mlp_kindling.md)
model spec. Parameters you want to search over are marked with
[`tune()`](https://hardhat.tidymodels.org/reference/tune.html).

``` r
spec = mlp_kindling(
    hidden_neurons = tune(),
    activations = tune(),
    epochs = 50,
    learn_rate = tune()
) |>
    set_mode("regression")
```

Note that `n_hlayer` is not listed here — it is handled inside
[`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md)
rather than the model spec directly.

### Data Preparation

We sample 30 rows per species to keep the example lightweight, and
stratify splits on `species` to preserve class balance. The target
variable is `body_mass_kg`, derived from the original `body_mass_g`
column.

``` r
penguins_clean = penguins |>
    drop_na() |>
    select(body_mass_g, ends_with("_mm"), sex, species) |>
    mutate(body_mass_kg = body_mass_g / 1000) |>
    slice_sample(n = 30, by = species)

set.seed(123)
split = initial_split(penguins_clean, prop = 0.8, strata = species)
train = training(split)
test = testing(split)
folds = vfold_cv(train, v = 5, strata = body_mass_kg)
```

    ## Warning: The number of observations in each quantile is below the recommended threshold
    ## of 20.
    ## • Stratification will use 3 breaks instead.

``` r
rec = recipe(body_mass_kg ~ ., data = train) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())
```

### Using grid_depth()

You still can use standard [dials](https://dials.tidymodels.org) grids
but the limitation is that they don’t know about network depth, so
[kindling](https://kindling.joshuamarie.com) provides
[`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md).
The `n_hlayer` argument controls which depths to search over. Remember,
it accepts:

- A scalar: `n_hlayer = 2`
- An integer vector: `n_hlayer = 1:3`
- A [dials](https://dials.tidymodels.org) range object:
  `n_hlayer = n_hlayer(c(1, 3))`

When `n_hlayer > 1`, the `hidden_neurons` and `activations` columns
become list-columns, where each row holds a vector of per-layer values.

``` r
set.seed(42)
depth_grid = grid_depth(
    hidden_neurons(c(16, 32)),
    activations(c("relu", "elu", "softshrink(lambd = 0.2)")),
    learn_rate(),
    n_hlayer = 1:3,
    size = 10,
    type = "latin_hypercube"
)

depth_grid
```

    ## # A tibble: 10 × 3
    ##    hidden_neurons activations learn_rate
    ##    <list>         <list>           <dbl>
    ##  1 <int [1]>      <chr [1]>     2.99e- 6
    ##  2 <int [2]>      <chr [2]>     9.46e- 5
    ##  3 <int [1]>      <chr [1]>     4.09e- 4
    ##  4 <int [1]>      <chr [1]>     2.98e- 8
    ##  5 <int [1]>      <chr [1]>     3.66e- 2
    ##  6 <int [3]>      <chr [3]>     1.62e- 7
    ##  7 <int [3]>      <chr [3]>     5.56e-10
    ##  8 <int [1]>      <chr [1]>     1.06e- 9
    ##  9 <int [1]>      <chr [1]>     1.40e- 5
    ## 10 <int [2]>      <chr [2]>     1.59e- 3

Here we constrain `hidden_neurons` to the range `[16, 32]` and limit
activations to three candidates — including the parametric `softshrink`.
Latin hypercube sampling spreads the 10 candidates more evenly across
the search space compared to a random grid.

### Tuning

What happens to the tuning part? The solution is easy: the parameters
induced into list-columns and it becomes something like `list(c(1, 2))`,
so internally the configured argument unlisted through
`list(c(1, 2))[[1]]` (it always produces only 1 element).

``` r
wflow = workflow() |>
    add_recipe(rec) |>
    add_model(spec)

tune_res = tune_grid(
    wflow,
    resamples = folds,
    grid = depth_grid,
    metrics = metric_set(rmse, rsq)
)
```

### Inspect

Even with the list-columns, it still normally produces the output we
want to produce. Use functions to extract the metrics output after grid
search,
e.g. [`collect_metrics()`](https://tune.tidymodels.org/reference/collect_predictions.html)
and
[`show_best()`](https://tune.tidymodels.org/reference/show_best.html).

``` r
collect_metrics(tune_res)
```

    ## # A tibble: 20 × 9
    ##    hidden_neurons activations learn_rate .metric .estimator  mean     n std_err
    ##    <list>         <list>           <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
    ##  1 <int [1]>      <chr [1]>     2.99e- 6 rmse    standard   2.64      5  0.0927
    ##  2 <int [1]>      <chr [1]>     2.99e- 6 rsq     standard   0.400     5  0.165 
    ##  3 <int [2]>      <chr [2]>     9.46e- 5 rmse    standard   1.84      5  0.232 
    ##  4 <int [2]>      <chr [2]>     9.46e- 5 rsq     standard   0.775     5  0.121 
    ##  5 <int [1]>      <chr [1]>     4.09e- 4 rmse    standard   3.29      5  0.155 
    ##  6 <int [1]>      <chr [1]>     4.09e- 4 rsq     standard   0.871     5  0.0287
    ##  7 <int [1]>      <chr [1]>     2.98e- 8 rmse    standard   3.15      5  0.0585
    ##  8 <int [1]>      <chr [1]>     2.98e- 8 rsq     standard   0.399     5  0.0958
    ##  9 <int [1]>      <chr [1]>     3.66e- 2 rmse    standard   3.30      5  0.0731
    ## 10 <int [1]>      <chr [1]>     3.66e- 2 rsq     standard   0.694     5  0.0862
    ## 11 <int [3]>      <chr [3]>     1.62e- 7 rmse    standard   0.424     5  0.0251
    ## 12 <int [3]>      <chr [3]>     1.62e- 7 rsq     standard   0.777     5  0.0422
    ## 13 <int [3]>      <chr [3]>     5.56e-10 rmse    standard   0.638     5  0.0692
    ## 14 <int [3]>      <chr [3]>     5.56e-10 rsq     standard   0.570     5  0.0609
    ## 15 <int [1]>      <chr [1]>     1.06e- 9 rmse    standard   3.44      5  0.0853
    ## 16 <int [1]>      <chr [1]>     1.06e- 9 rsq     standard   0.832     5  0.0870
    ## 17 <int [1]>      <chr [1]>     1.40e- 5 rmse    standard   3.51      5  0.164 
    ## 18 <int [1]>      <chr [1]>     1.40e- 5 rsq     standard   0.516     5  0.192 
    ## 19 <int [2]>      <chr [2]>     1.59e- 3 rmse    standard   2.13      5  0.0803
    ## 20 <int [2]>      <chr [2]>     1.59e- 3 rsq     standard   0.684     5  0.102 
    ## # ℹ 1 more variable: .config <chr>

``` r
show_best(tune_res, metric = "rmse", n = 5)
```

    ## # A tibble: 5 × 9
    ##   hidden_neurons activations learn_rate .metric .estimator  mean     n std_err
    ##   <list>         <list>           <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
    ## 1 <int [3]>      <chr [3]>     1.62e- 7 rmse    standard   0.424     5  0.0251
    ## 2 <int [3]>      <chr [3]>     5.56e-10 rmse    standard   0.638     5  0.0692
    ## 3 <int [2]>      <chr [2]>     9.46e- 5 rmse    standard   1.84      5  0.232 
    ## 4 <int [2]>      <chr [2]>     1.59e- 3 rmse    standard   2.13      5  0.0803
    ## 5 <int [1]>      <chr [1]>     2.99e- 6 rmse    standard   2.64      5  0.0927
    ## # ℹ 1 more variable: .config <chr>

## Visualizing Results

## Finalizing the Model

Once we’ve identified the best configuration, we finalize the workflow
and fit it on the full training set.

``` r
best_params = select_best(tune_res, metric = "rmse")
final_wflow = wflow |>
    finalize_workflow(best_params)

final_model = fit(final_wflow, data = train)
final_model
```

    ## ══ Workflow [trained] ══════════════════════════════════════════════════════════
    ## Preprocessor: Recipe
    ## Model: mlp_kindling()
    ## 
    ## ── Preprocessor ────────────────────────────────────────────────────────────────
    ## 2 Recipe Steps
    ## 
    ## • step_dummy()
    ## • step_normalize()
    ## 
    ## ── Model ───────────────────────────────────────────────────────────────────────

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2
    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## 
    ## ======================= Feedforward Neural Networks (MLP) ======================
    ## 
    ## 
    ## -- FFNN Model Summary ----------------------------------------------------------

    ## Warning in system("tput cols", intern = TRUE): running command 'tput cols' had
    ## status 2

    ## -------------------------------------------------------------------
    ##   NN Model Type           :         FFNN    n_predictors :      7
    ##   Number of Epochs        :           50    n_response   :      1
    ##   Hidden Layer Units      :   31, 32, 32    reg.         :   None
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
    ##   1st Layer {31}    :                      relu
    ##   2nd Layer {32}    :                       elu
    ##   3rd Layer {32}    :   softshrink(lambd = 0.2)
    ##   Output Activation :   No act function applied
    ## -------------------------------------------------

### Evaluating on the test set

``` r
final_model |>
    augment(new_data = test) |>
    metric_set(rmse, rsq)(
        truth = body_mass_kg,
        estimate = .pred
    )
```

    ## # A tibble: 2 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       0.532
    ## 2 rsq     standard       0.545

## A Note on Parametric Activations

[kindling](https://kindling.joshuamarie.com) supports parametric
activation functions, meaning each layer’s activation can carry its own
tunable parameter. When passed as a string such as
`"softshrink(lambd = 0.2)"`,
[kindling](https://kindling.joshuamarie.com) parses and constructs the
activation automatically. This means you can include them directly in
the
[`activations()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md)
candidate list inside
[`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md)
without any extra setup, as shown above.

For manual (non-tuned) use, you can also specify activations per layer
explicitly:

``` r
spec_manual = mlp_kindling(
    hidden_neurons = c(50, 15),
    activations = act_funs(
        softshrink[lambd = 0.5],
        relu
    ),
    epochs = 150,
    learn_rate = 0.01
) |>
    set_mode("regression")
```
