test_that("mlp_kindling works within workflows", {
    skip_if_not_installed("workflows")
    skip_if_not_installed("recipes")
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = c(10),
        epochs = 5,
        verbose = FALSE
    )

    rec = recipes::recipe(Species ~ ., data = iris)

    wf = workflows::workflow() |>
        workflows::add_recipe(rec) |>
        workflows::add_model(spec)

    expect_s3_class(wf, "workflow")

    fitted_wf = workflows::fit(wf, data = iris[1:100, ])

    expect_s3_class(fitted_wf, "workflow")

    preds = predict(fitted_wf, new_data = iris[101:110, ])

    expect_s3_class(preds, "tbl_df")
    expect_equal(nrow(preds), 10)
})

test_that("rnn_kindling works within workflows", {
    skip_if_not_installed("workflows")
    skip_if_not_installed("recipes")
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = rnn_kindling(
        mode = "classification",
        hidden_neurons = c(10),
        rnn_type = "gru",
        epochs = 5,
        verbose = FALSE
    )

    rec = recipes::recipe(Species ~ ., data = iris)

    wf = workflows::workflow() |>
        workflows::add_recipe(rec) |>
        workflows::add_model(spec)

    expect_s3_class(wf, "workflow")

    fitted_wf = workflows::fit(wf, data = iris[1:100, ])

    expect_s3_class(fitted_wf, "workflow")
})

test_that("workflow with preprocessing recipe works", {
    skip_if_not_installed("workflows")
    skip_if_not_installed("recipes")
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "regression",
        hidden_neurons = c(10),
        epochs = 5,
        verbose = FALSE
    )

    rec = recipes::recipe(Sepal.Length ~ ., data = iris) |>
        recipes::step_normalize(recipes::all_numeric_predictors()) |>
        recipes::step_dummy(recipes::all_factor_predictors())

    wf = workflows::workflow() |>
        workflows::add_recipe(rec) |>
        workflows::add_model(spec)

    fitted_wf = workflows::fit(wf, data = iris[1:100, ])
    preds = predict(fitted_wf, new_data = iris[101:110, ])

    expect_s3_class(preds, "tbl_df")
    expect_equal(nrow(preds), 10)
    expect_true(all(is.finite(preds$.pred)))
})
