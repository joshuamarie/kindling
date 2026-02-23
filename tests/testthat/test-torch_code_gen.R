skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("nn_module_generator() errors when no_x is missing", {
    skip_if_no_torch()
    expect_error(nn_module_generator(hd_neurons = 4, no_y = 2))
})

test_that("nn_module_generator() errors when no_y is missing", {
    skip_if_no_torch()
    expect_error(nn_module_generator(hd_neurons = 4, no_x = 3))
})

test_that("nn_module_generator() returns a quosure when eval = FALSE", {
    skip_if_no_torch()
    out = nn_module_generator(hd_neurons = 8, no_x = 4, no_y = 1)
    expect_true(rlang::is_quosure(out))
})

test_that("nn_module_generator() returns nn_module class when eval = TRUE", {
    skip_if_no_torch()
    out = nn_module_generator(hd_neurons = 8, no_x = 4, no_y = 1, eval = TRUE)
    expect_true(inherits(out, "nn_module_generator"))
})

test_that("nn_module_generator() works with no hidden layers", {
    skip_if_no_torch()
    expect_no_error(nn_module_generator(hd_neurons = c(), no_x = 4, no_y = 1))
    expect_no_error(nn_module_generator(hd_neurons = NULL, no_x = 4, no_y = 1))
    expect_no_error(nn_module_generator(no_x = 4, no_y = 1))
})


