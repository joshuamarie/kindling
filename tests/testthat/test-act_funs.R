skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("args() returns activation_args class", {
    skip_if_no_torch()
    
    expect_warning({ res = kindling::args(dim = 2L) }, class = "lifecycle_warning_deprecated")
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

test_that(".assert_tensor_output() errors on non-tensor output", {
    skip_if_no_torch()
    expect_error(
        new_act_fn(\(x) as.numeric(x))(),
        class = "custom_activation_output_error"
    )
})

test_that("new_act_fn() call-time guard errors on non-tensor", {
    skip_if_no_torch()
    fn = new_act_fn(\(x) x, probe = FALSE)
    guarded = attr(fn, "act_fn")
    expect_error(
        guarded(42),
        class = "custom_activation_output_error"
    )
})

# ---- eval_act_funs() ----

test_that("eval_act_funs() returns NULL for both when not supplied", {
    res = eval_act_funs(NULL, NULL)
    expect_null(res$activations)
    expect_null(res$output_activation)
})

test_that("eval_act_funs() passes through act_funs() spec", {
    skip_if_no_torch()
    res = eval_act_funs(act_funs(relu), NULL)
    expect_s3_class(res$activations, "activation_spec")
    expect_null(res$output_activation)
})

test_that("eval_act_funs() handles output_activation", {
    skip_if_no_torch()
    res = eval_act_funs(NULL, act_funs(relu))
    expect_null(res$activations)
    expect_s3_class(res$output_activation, "activation_spec")
})
