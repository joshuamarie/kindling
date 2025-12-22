test_that("mlp_kindling is properly registered with parsnip", {
    skip_if_not_installed("parsnip")

    models = parsnip::get_model_env()$models

    expect_true("mlp_kindling" %in% models)
})

test_that("rnn_kindling is properly registered with parsnip", {
    skip_if_not_installed("parsnip")

    models = parsnip::get_model_env()$models

    expect_true("rnn_kindling" %in% models)
})

test_that("mlp_kindling supports both regression and classification modes", {
    skip_if_not_installed("parsnip")

    spec_class = mlp_kindling(mode = "classification")
    spec_reg = mlp_kindling(mode = "regression")

    expect_equal(spec_class$mode, "classification")
    expect_equal(spec_reg$mode, "regression")
})

test_that("rnn_kindling supports both regression and classification modes", {
    skip_if_not_installed("parsnip")

    spec_class = rnn_kindling(mode = "classification")
    spec_reg = rnn_kindling(mode = "regression")

    expect_equal(spec_class$mode, "classification")
    expect_equal(spec_reg$mode, "regression")
})
