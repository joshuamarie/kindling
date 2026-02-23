skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("act_funs() accepts bare symbols", {
    skip_if_no_torch()
    
    spec = act_funs(relu, elu)
    expect_s3_class(spec, "activation_spec")
    expect_length(spec, 2)
})

test_that("act_funs() accepts character strings", {
    skip_if_no_torch()
    
    spec = act_funs("relu", "elu")
    expect_s3_class(spec, "activation_spec")
    expect_length(spec, 2)
})

test_that("act_funs() accepts parameterized string", {
    skip_if_no_torch()
    
    spec = act_funs("softshrink(lambd = 0.1)")
    elem = spec[[1]]
    expect_s3_class(elem, "parameterized_activation")
    expect_equal(attr(elem, "act_name"), "softshrink")
})

test_that("act_funs() accepts indexed bracket syntax with named param", {
    skip_if_no_torch()
    
    spec = act_funs(softshrink[lambd = 0.2])
    elem = spec[[1]]
    expect_s3_class(elem, "parameterized_activation")
    expect_equal(attr(elem, "act_name"), "softshrink")
    expect_equal(elem$lambd, 0.2)
})

test_that("act_funs() accepts indexed bracket syntax with unnamed param", {
    skip_if_no_torch()
    
    spec = act_funs(softshrink[0.5])
    elem = spec[[1]]
    expect_s3_class(elem, "parameterized_activation")
    expect_equal(attr(elem, "act_name"), "softshrink")
})

test_that("act_funs() returns activation_spec of correct length", {
    skip_if_no_torch()
    
    spec = act_funs(relu, elu, sigmoid)
    expect_length(spec, 3)
})

test_that("act_funs() errors on invalid activation name", {
    skip_if_no_torch()
    
    expect_error(act_funs(not_a_real_fn), class = "activation_not_found_error")
})

test_that("act_funs() errors on invalid parameter name", {
    skip_if_no_torch()
    
    expect_error(act_funs(relu[invalid_param = 1]), class = "purrr_error_indexed")
})

test_that("act_funs() errors on invalid syntax", {
    skip_if_no_torch()
    
    expect_error(act_funs(123), class = "activation_syntax_error")
})

test_that("args() returns activation_args class", {
    skip_if_no_torch()
    
    expect_warning({ res = args(dim = 2L) }, class = "lifecycle_warning_deprecated")
    expect_s3_class(res, "activation_args")
})

test_that("args() is deprecated", {
    skip_if_no_torch()
    
    expect_warning(args(dim = 2L), class = "lifecycle_warning_deprecated")
})

test_that("args() errors on unnamed params", {
    skip_if_no_torch()
    
    expect_warning(
        expect_error(args(1L), class = "activation_args_error"),
        class = "lifecycle_warning_deprecated"
    )
})

# ---- Custom activation function ----

test_that("new_act_fn() accepts a valid function", {
    skip_if_no_torch()
    fn = new_act_fn(\(x) torch::torch_tanh(x))
    expect_s3_class(fn, "custom_activation")
    expect_s3_class(fn, "parameterized_activation")
})

test_that("new_act_fn() errors if fn is not a function", {
    expect_error(new_act_fn("not a fn"), class = "custom_activation_type_error")
})

test_that("new_act_fn() errors if fn has no arguments", {
    expect_error(new_act_fn(function() 1), class = "custom_activation_arity_error")
})

# test_that("new_act_fn() errors if fn does not return a tensor", {
#     skip_if_no_torch()
#     expect_error(
#         new_act_fn(\(x) as.numeric(x)),
#         class = "custom_activation_probe_error"
#     )
# })

test_that("new_act_fn() works inside act_funs()", {
    skip_if_no_torch()
    spec = act_funs(relu, new_act_fn(\(x) torch::torch_tanh(x)))
    expect_length(spec, 2)
    expect_s3_class(spec[[2]], "custom_activation")
})

test_that("new_act_fn() skips probe when probe = FALSE", {
    fn = new_act_fn(\(x) x, probe = FALSE)
    expect_s3_class(fn, "custom_activation")
})

test_that("new_act_fn() stores custom .name attribute", {
    skip_if_no_torch()
    fn = new_act_fn(\(x) torch::torch_tanh(x), .name = "my_tanh")
    expect_equal(attr(fn, "act_name"), "my_tanh")
})
