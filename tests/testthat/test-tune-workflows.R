skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("Tuning mlp_kindling with grid_depth works", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("workflows")
    skip_if_not_installed("tune")
    skip_if_not_installed("rsample")
    skip_if_not_installed("recipes")
    skip_if_not_installed("dials")
    skip_if_no_torch()

    mlp_spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = tune::tune(),
        activations = tune::tune(),
        output_activation = tune::tune(),
        epochs = 10
    )

    iris_recipe = recipes::recipe(Species ~ ., data = iris)
    wf = workflows::workflow() |>
        workflows::add_recipe(iris_recipe) |>
        workflows::add_model(mlp_spec)

    set.seed(123)
    folds = rsample::vfold_cv(iris, v = 2)

    grid = grid_depth(
        hidden_neurons(c(16L, 32L)),
        activations(c("relu", "elu")),
        output_activation(c("sigmoid", "linear")),
        n_hlayer = 2,
        size = 3,
        type = "random"
    )

    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 3)
    expect_true("hidden_neurons" %in% names(grid))
    expect_true("activations" %in% names(grid))
    expect_true("output_activation" %in% names(grid))
    expect_true(is.list(grid$hidden_neurons))
    expect_true(is.list(grid$activations))
    expect_equal(length(grid$hidden_neurons[[1]]), 2)
    expect_equal(length(grid$activations[[1]]), 2)

    out = tune::tune_grid(
        wf,
        resamples = folds,
        grid = grid,
        control = tune::control_grid(save_pred = FALSE)
    )

    expect_s3_class(out, "tune_results")
    expect_true(nrow(tune::collect_metrics(out)) > 0)
})

test_that("Tuning mlp_kindling with latin_hypercube grid works", {
    skip_if_not_installed("lhs")
    skip_if_not_installed("parsnip")
    skip_if_not_installed("workflows")
    skip_if_not_installed("tune")
    skip_if_no_torch()

    mlp_spec = mlp_kindling(
        mode = "regression",
        hidden_neurons = tune::tune(),
        activations = tune::tune(),
        epochs = 5
    )

    mtcars_recipe = recipes::recipe(mpg ~ ., data = mtcars)
    wf = workflows::workflow() |>
        workflows::add_recipe(mtcars_recipe) |>
        workflows::add_model(mlp_spec)

    set.seed(456)
    folds = rsample::vfold_cv(mtcars, v = 2)

    grid = grid_depth(
        hidden_neurons(c(8L, 32L)),
        activations(c("relu", "elu")),
        n_hlayer = 2:3,
        size = 4,
        type = "latin_hypercube"
    )

    expect_equal(nrow(grid), 4)
    expect_true(all(purrr::map_int(grid$hidden_neurons, length) %in% 2:3))

    results = tune::tune_grid(
        wf,
        resamples = folds,
        grid = grid,
        control = tune::control_grid(save_pred = FALSE)
    )

    expect_s3_class(results, "tune_results")

    # ---SELECT BEST---
    best = tune::select_best(results, metric = "rmse")
    expect_s3_class(best, "tbl_df")
    expect_true("hidden_neurons" %in% names(best))
})

test_that("Tuning rnn_kindling with grid_depth works", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("workflows")
    skip_if_not_installed("tune")
    skip_if_not_installed("rsample")
    skip_if_not_installed("recipes")
    skip_if_no_torch()

    rnn_spec = rnn_kindling(
        mode = "classification",
        hidden_neurons = tune::tune(),
        activations = tune::tune(),
        epochs = 5
    )

    iris_recipe = recipes::recipe(Species ~ ., data = iris)
    wf = workflows::workflow() |>
        workflows::add_recipe(iris_recipe) |>
        workflows::add_model(rnn_spec)

    set.seed(789)
    folds = rsample::vfold_cv(iris, v = 2)

    grid = grid_depth(
        hidden_neurons(c(16L, 32L)),
        activations(c("relu", "elu")),
        n_hlayer = 2,
        size = 3,
        type = "random"
    )

    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 3)
    expect_true("hidden_neurons" %in% names(grid))
    expect_true("activations" %in% names(grid))

    # --RNN type shouldn't be tunable, thus it won't exist
    expect_false("rnn_type" %in% names(grid))

    OUT = tune::tune_grid(
        wf,
        resamples = folds,
        grid = grid,
        control = tune::control_grid(save_pred = FALSE)
    )

    expect_s3_class(OUT, "tune_results")
    expect_true(nrow(tune::collect_metrics(OUT)) > 0)
})

test_that("grid_depth handles different n_hlayer specifications", {
    skip_if_no_torch()

    grid1 = grid_depth(
        hidden_neurons(c(16L, 32L)),
        activations(c("relu", "elu")),
        n_hlayer = 2,
        size = 5,
        type = "random"
    )

    expect_true(any(purrr::map_int(grid1$hidden_neurons, length) == 2))
    expect_true(any(purrr::map_int(grid1$activations, length) == 2))

    # ---Multiple depths---
    grid2 = grid_depth(
        hidden_neurons(c(16L, 32L)),
        activations(c("relu", "elu")),
        n_hlayer = 1:3,
        size = 9,
        type = "random"
    )

    depths = purrr::map_int(grid2$hidden_neurons, length)
    expect_true(all(depths %in% 1:3))
})

test_that("grid_depth works with workflow method", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("workflows")
    skip_if_no_torch()

    mlp_spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = tune::tune(),
        activations = tune::tune(),
        epochs = tune::tune()
    )

    wf = workflows::workflow() |>
        workflows::add_recipe(recipes::recipe(Species ~ ., data = iris)) |>
        workflows::add_model(mlp_spec)

    grid = grid_depth(
        wf,
        n_hlayer = 2,
        size = 5,
        type = "random"
    )

    expect_s3_class(grid, "tbl_df")
    expect_true("hidden_neurons" %in% names(grid))
    expect_true("activations" %in% names(grid))
    expect_true("epochs" %in% names(grid))
})

test_that("grid_depth handles scalar parameters correctly", {
    skip_if_no_torch()

    grid = grid_depth(
        hidden_neurons(c(16L, 32L)),
        activations(c("relu", "elu")),

        # `epochs` belong to `{dials}`, not in `{kindling}`
        dials::epochs(c(10L, 50L)),

        # `learn_rate` belong to `{dials}`, not in `{kindling}`
        dials::learn_rate(c(0.001, 0.1)),

        n_hlayer = 2,
        size = 5,
        type = "random"
    )

    expect_equal(nrow(grid), 5)
    expect_true("epochs" %in% names(grid))
    expect_true("learn_rate" %in% names(grid))
})

test_that("finalize_workflow works with grid_depth results", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("workflows")
    skip_if_not_installed("tune")
    skip_if_not_installed("rsample")
    skip_if_no_torch()

    mlp_spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = tune::tune(),
        activations = tune::tune(),
        epochs = 10
    )

    wf = workflows::workflow() |>
        workflows::add_recipe(recipes::recipe(Species ~ ., data = iris)) |>
        workflows::add_model(mlp_spec)

    set.seed(321)
    folds = rsample::vfold_cv(iris, v = 2)

    grid = grid_depth(
        hidden_neurons(c(16L, 32L)),
        activations(c("relu", "elu")),
        n_hlayer = 2,
        size = 3,
        type = "random"
    )

    results = tune::tune_grid(wf, folds, grid = grid)
    best = tune::select_best(results, metric = "accuracy")

    final_wf = tune::finalize_workflow(wf, best)

    expect_s3_class(final_wf, "workflow")

    # ---Final parameters stores into a list—expect no failures---
    testthat::expect_no_failure({
        final_nn_model = parsnip::fit(final_wf, data = iris)
        final_nn_model
    })
})
