skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

# ---- grid_depth() ----

test_that("grid_depth() works with param objects (grid_depth.param)", {
    grid = grid_depth(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu")),
        n_hlayer = 2L,
        type = "random",
        size = 5
    )
    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 5)
    expect_true("hidden_neurons" %in% names(grid))
    expect_true("activations" %in% names(grid))
    expect_true(all(lengths(grid$hidden_neurons) == 2L))
})

test_that("grid_depth() with n_hlayer range produces variable-length list cols", {
    grid = grid_depth(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu")),
        n_hlayer = 2:3,
        type = "random",
        size = 20
    )
    depths = lengths(grid$hidden_neurons)
    expect_true(all(depths %in% 2:3))
})

test_that("grid_depth() type = 'regular' works", {
    grid = grid_depth(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu")),
        n_hlayer = 2L,
        type = "regular",
        levels = 2
    )
    expect_s3_class(grid, "tbl_df")
    expect_gt(nrow(grid), 0)
})

test_that("grid_depth() type = 'latin_hypercube' works", {
    skip_if_not_installed("lhs")
    skip_if_no_torch()
    grid = grid_depth(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu")),
        dials::epochs(c(50L, 200L)),
        n_hlayer = 2L,
        type = "latin_hypercube",
        size = 8
    )
    expect_equal(nrow(grid), 8)
    expect_true("epochs" %in% names(grid))
})

test_that("grid_depth() works from a list", {
    params = list(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu"))
    )
    grid = grid_depth(params, n_hlayer = 2L, type = "random", size = 5)
    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 5)
})

test_that("grid_depth() works from a parameters object", {
    params = dials::parameters(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu"))
    )
    grid = grid_depth(params, n_hlayer = 2L, type = "random", size = 5)
    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 5)
})

test_that("grid_depth() with n_hlayer = 1 returns scalar columns", {
    grid = grid_depth(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu")),
        n_hlayer = 1L,
        type = "random",
        size = 5
    )
    expect_type(grid$hidden_neurons, "integer")
    expect_type(grid$activations, "character")
})

test_that("grid_depth() with n_hlayer as param object works", {
    grid = grid_depth(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu")),
        n_hlayer = n_hlayers(range = c(2L, 3L)),
        type = "random",
        size = 10
    )
    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 10)
})

test_that("grid_depth() errors on unsupported class", {
    expect_error(grid_depth(42), class = "rlang_error")
})

test_that("grid_depth() works from a workflow", {
    skip_if_not_installed("workflows")
    skip_if_not_installed("tune")
    skip_if_no_torch()
    
    wf = workflows::workflow() |>
        workflows::add_model(
            mlp_kindling(
                hidden_neurons = tune::tune(), 
                activations = tune::tune(),
                mode = "regression"
            )
        ) |>
        workflows::add_formula(Sepal.Length ~ .)
    
    grid = grid_depth(wf, hidden_neurons(c(32L, 128L)), n_hlayer = 2L, type = "random", size = 5)
    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 5)
})

# ---- dials parameter constructors ----

test_that("hidden_neurons() with disc_values restricts to those values", {
    vals = c(32L, 64L, 128L)
    p = hidden_neurons(disc_values = vals)
    grid = dials::grid_random(p, size = 20)
    expect_true(all(grid$hidden_neurons %in% vals))
})

test_that("hidden_neurons() errors on non-positive disc_values", {
    expect_error(hidden_neurons(disc_values = c(-1L, 64L)), class = "rlang_error")
})

test_that("hidden_neurons() errors on NA in disc_values", {
    expect_error(hidden_neurons(disc_values = c(32L, NA_integer_)), class = "rlang_error")
})

test_that("hidden_neurons() with disc_values works inside grid_depth()", {
    grid = grid_depth(
        hidden_neurons(disc_values = c(32L, 64L, 128L)),
        activations(c("relu", "elu")),
        n_hlayer = 2L,
        type = "random",
        size = 5
    )
    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 5)
})

test_that("output_activation() values are usable in a grid", {
    grid = dials::grid_random(
        output_activation(c("relu", "softmax")),
        size = 4
    )
    expect_true(all(grid$output_activation %in% c("relu", "softmax")))
})

test_that("optimizer() values are usable in a grid", {
    grid = dials::grid_random(
        optimizer(c("adam", "sgd", "adamw")),
        size = 6
    )
    expect_true(all(grid$optimizer %in% c("adam", "sgd", "adamw")))
})

test_that("bias() samples only TRUE/FALSE", {
    grid = dials::grid_random(bias(), size = 10)
    expect_true(all(grid$bias %in% c(TRUE, FALSE)))
})

test_that("validation_split() stays within range", {
    grid = dials::grid_random(validation_split(range = c(0.1, 0.3)), size = 10)
    expect_true(all(grid$validation_split >= 0.1 & grid$validation_split <= 0.3))
})

test_that("bidirectional() samples only TRUE/FALSE", {
    grid = dials::grid_random(bidirectional(), size = 10)
    expect_true(all(grid$bidirectional %in% c(TRUE, FALSE)))
})

test_that("n_hlayers() respects custom range", {
    p = n_hlayers(range = c(1L, 5L))
    r = dials::range_get(p)
    expect_equal(r$lower, 1L)
    expect_equal(r$upper, 5L)
})
