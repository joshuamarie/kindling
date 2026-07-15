# ---- parse_activation_string() ----

test_that("parse_activation_string handles a bare name with no parens", {
    out = parse_activation_string("relu")
    expect_equal(out$name, "relu")
    expect_equal(out$params, list())
})

test_that("parse_activation_string rejects non-character or length != 1 input", {
    expect_error(parse_activation_string(123))
    expect_error(parse_activation_string(c("relu", "tanh")))
})

test_that("parse_activation_string errors on an unbalanced/invalid paren format", {
    expect_error(parse_activation_string("foo(x=1"), "Invalid activation string format")
})

test_that("parse_activation_string handles empty parens as no params", {
    out = parse_activation_string("foo()")
    expect_equal(out$name, "foo")
    expect_equal(out$params, list())
})

test_that("parse_activation_string parses named params inside parens", {
    out = parse_activation_string("softshrink(lambd = 0.1)")
    expect_equal(out$name, "softshrink")
    expect_equal(out$params, list(lambd = 0.1))
})

test_that("parse_activation_string surfaces a clear error on unparseable params", {
    expect_error(parse_activation_string("foo(x=1,)"), "Failed to parse parameters")
})

# ---- parse_activation_spec() ----

test_that("parse_activation_spec returns NA names/empty params for NULL input", {
    out = parse_activation_spec(NULL, n_layers = 3)
    expect_equal(out$names, rep(NA_character_, 3))
    expect_equal(length(out$params), 3)
})

test_that("parse_activation_spec recycles and errors on length mismatch for activation_spec objects", {
    spec1 = structure(list("relu"), class = "activation_spec")
    out = parse_activation_spec(spec1, n_layers = 3)
    expect_equal(length(out$names), 3)

    spec2 = structure(list("relu", "tanh"), class = "activation_spec")
    expect_error(parse_activation_spec(spec2, n_layers = 3), class = "activation_spec_length_error")
})

test_that("parse_activation_spec handles custom_activation and parameterized_activation elements", {
    custom_elem = structure(list(), act_fn = function(x) x, class = "custom_activation")
    param_elem = structure(list(lambd = 0.2), act_name = "softshrink", class = "parameterized_activation")
    spec = structure(list(custom_elem, param_elem), class = "activation_spec")

    out = parse_activation_spec(spec, n_layers = 2)
    expect_equal(out$names[[1]], "<custom>")
    expect_equal(out$names[[2]], "softshrink")
    expect_equal(out$params[[2]]$lambd, 0.2)
})

test_that("parse_activation_spec errors on an unsupported element type inside an activation_spec", {
    bad_spec = structure(list(42), class = "activation_spec")
    expect_error(parse_activation_spec(bad_spec, n_layers = 1))
})

test_that("parse_activation_spec recycles a length-1 character vector and errors on mismatch", {
    out = parse_activation_spec("relu", n_layers = 3)
    expect_equal(unlist(out$names), rep("relu", 3))

    expect_error(parse_activation_spec(c("relu", "tanh"), n_layers = 3), class = "activation_spec_length_error")
})

test_that("parse_activation_spec parses a character vector via parse_activation_string", {
    out = parse_activation_spec(c("relu", "softshrink(lambd = 0.1)"), n_layers = 2)
    expect_equal(out$names[[1]], "relu")
    expect_equal(out$names[[2]], "softshrink")
    expect_equal(out$params[[2]]$lambd, 0.1)
})

test_that("parse_activation_spec handles the documented mixed named/unnamed example", {
    # this is the exact syntax from the roxygen docs: list(relu, tanh, softmax = args(dim = 2L))
    out = parse_activation_spec(list("relu", "tanh", softmax = list(dim = 2L)), n_layers = 3)
    expect_equal(unlist(out$names), c("relu", "tanh", "softmax"))
    expect_equal(out$params[[3]]$dim, 2L)
})

test_that("parse_activation_spec handles list input: named list, named empty string, unnamed string", {
    out = parse_activation_spec(
        list(softmax = list(dim = 2L), other = "", "relu"),
        n_layers = 3
    )
    expect_equal(out$names[[1]], "softmax")
    expect_equal(out$params[[1]]$dim, 2L)
    expect_equal(out$names[[2]], "other")
    expect_equal(out$params[[2]], list())
    expect_equal(out$names[[3]], "relu")
})

test_that("parse_activation_spec recycles a length-1 list and errors on length mismatch", {
    out = parse_activation_spec(list("relu"), n_layers = 2)
    expect_equal(length(out$names), 2)

    expect_error(parse_activation_spec(list("relu", "tanh"), n_layers = 3), class = "activation_spec_length_error")
})

test_that("parse_activation_spec errors on an invalid unnamed, non-character list element", {
    expect_error(parse_activation_spec(list(42), n_layers = 1), class = "activation_syntax_error")
})

test_that("parse_activation_spec errors for an unsupported top-level activations type", {
    expect_error(parse_activation_spec(42, n_layers = 2), class = "activation_type_error")
})

# ---- act_funs(): superseded named = args(...) DSL branch ----

test_that("act_funs() accepts the superseded named = args(...) syntax", {
    skip_if_not_installed("torch")
    withr::local_options(lifecycle_verbosity = "quiet")

    out = act_funs(softshrink = args(lambd = 0.5))

    expect_s3_class(out, "activation_spec")
    expect_true(inherits(out[[1]], "parameterized_activation"))
    expect_equal(attr(out[[1]], "act_name"), "softshrink")
    expect_equal(unclass(out[[1]])$lambd, 0.5)
})

test_that("act_funs() accepts named = \"\" as a parameterless named activation", {
    skip_if_not_installed("torch")

    out = act_funs(relu = "")

    expect_true(inherits(out[[1]], "parameterized_activation"))
    expect_equal(attr(out[[1]], "act_name"), "relu")
    expect_length(unclass(out[[1]]), 0)
})

test_that("act_funs() errors on invalid named-parameter syntax", {
    skip_if_not_installed("torch")

    expect_error(
        act_funs(relu = 42),
        class = "activation_syntax_error"
    )
})

test_that("act_funs() rejects unrecognized call expressions", {
    skip_if_not_installed("torch")

    expect_error(
        act_funs(some_undefined_call(1, 2)),
        class = "activation_syntax_error"
    )
})

# ---- args() ----
# This will be hard deprecated soon
# Possibly removed

test_that("args() returns an empty activation_args object when called with no arguments", {
    withr::local_options(lifecycle_verbosity = "quiet")

    out = args()
    expect_s3_class(out, "activation_args")
    expect_equal(unclass(out), list())
})

test_that("args() requires all arguments to be named", {
    withr::local_options(lifecycle_verbosity = "quiet")

    expect_error(
        args(0.5),
        class = "activation_args_error"
    )
})

test_that("args() is soft-deprecated in favor of indexed syntax", {
    withr::local_options(lifecycle_verbosity = "warning")

    expect_warning(
        args(lambd = 0.5),
        class = "lifecycle_warning_deprecated"
    )
})

# ---- validate_activation() ----

test_that("validate_activation() treats 'linear' as a reserved no-op activation", {
    skip_if_not_installed("torch")
    expect_equal(validate_activation("linear"), "linear")
})

test_that("validate_activation() errors when the torch function doesn't exist", {
    skip_if_not_installed("torch")
    expect_error(
        validate_activation("definitely_not_a_real_activation"),
        class = "activation_not_found_error"
    )
})

test_that("validate_activation() errors when torch isn't installed", {
    testthat::local_mocked_bindings(
        is_installed = function(...) FALSE,
        .package = "kindling"
    )
    expect_error(
        validate_activation("relu"),
        class = "torch_missing_error"
    )
})

# ---- validate_args_formals() ----

test_that("validate_args_formals() allows 'linear' only with zero parameters", {
    skip_if_not_installed("torch")
    expect_null(validate_args_formals("linear", list()))
    expect_error(
        validate_args_formals("linear", list(foo = 1)),
        class = "activation_invalid_params_error"
    )
})

test_that("validate_args_formals() returns early when torch isn't installed", {
    testthat::local_mocked_bindings(
        has_namespace = function(...) FALSE,
        .package = "kindling"
    )
    expect_null(validate_args_formals("relu", list(foo = 1)))
})

test_that("validate_args_formals() errors on parameter names not in the function's formals", {
    skip_if_not_installed("torch")
    expect_error(
        validate_args_formals("softshrink", list(not_a_real_arg = 1)),
        class = "activation_invalid_params_error"
    )
})

# ---- process_activations() ----

test_that("process_activations() emits NULL for NA names", {
    spec = list(names = NA_character_, params = list(list()))
    fns = process_activations(spec)
    expect_null(fns[[1]])
})

test_that("process_activations() emits identity() for 'linear'", {
    spec = list(names = "linear", params = list(list()))
    fns = process_activations(spec)
    expect_equal(fns[[1]](quote(x)), quote(identity(x)))
})

test_that("process_activations() dispatches custom activation functions", {
    my_fn = function(x) x^2
    spec = list(names = "<custom>", params = list(list(fn = my_fn)))
    fns = process_activations(spec)
    expect_equal(fns[[1]](quote(x)), rlang::expr((!!my_fn)(x)))
})

test_that("process_activations() builds a namespaced torch call with no params", {
    spec = list(names = "relu", params = list(list()))
    fns = process_activations(spec)
    expect_equal(fns[[1]](quote(x)), quote(torch::nnf_relu(x)))
})

test_that("process_activations() builds a namespaced torch call with params", {
    spec = list(names = "softshrink", params = list(list(lambd = 0.5)))
    fns = process_activations(spec)
    expect_equal(fns[[1]](quote(x)), quote(torch::nnf_softshrink(x, lambd = 0.5)))
})

# ---- eval_act_funs() ----

test_that("eval_act_funs() returns NULL for both fields when neither is supplied", {
    out = eval_act_funs(activations = NULL, output_activation = NULL)
    expect_null(out$activations)
    expect_null(out$output_activation)
})

test_that("eval_act_funs() evaluates activations/output_activation within the act_funs()/args() DSL mask", {
    skip_if_not_installed("torch")
    withr::local_options(lifecycle_verbosity = "quiet")

    out = eval_act_funs(
        activations = act_funs(relu, sigmoid),
        output_activation = act_funs(softmax = args(dim = 2L))
    )

    expect_s3_class(out$activations, "activation_spec")
    expect_s3_class(out$output_activation, "activation_spec")
})
