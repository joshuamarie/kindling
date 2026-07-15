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
