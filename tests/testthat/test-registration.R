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

test_that("make_kindling() fully registers all three models with parsnip", {
    model_env = parsnip::get_model_env()

    # force past the "already registered" guard so the traced call below
    # actually executes the real registration body, not the early return
    for (m in c("mlp_kindling", "rnn_kindling", "train_nnsnip")) {
        if (m %in% model_env$models) {
            if (exists(m, envir = model_env)) rm(list = m, envir = model_env)
            model_env$models = setdiff(model_env$models, m)
        }
    }

    make_kindling()

    expect_true("mlp_kindling" %in% model_env$models)
    expect_true("rnn_kindling" %in% model_env$models)
    expect_true("train_nnsnip" %in% model_env$models)
})

test_that("make_kindling() takes the early-return guard on a second call", {
    expect_true(make_kindling())
})

