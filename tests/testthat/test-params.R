# ---- dials-kindling.R: parameter constructors ----

test_that("hidden_neurons works with range and with disc_values", {
    p1 = hidden_neurons(range = c(8L, 64L))
    expect_s3_class(p1, "quant_param")

    p2 = hidden_neurons(disc_values = c(16L, 32L, 64L))
    expect_s3_class(p2, "quant_param")

    expect_error(hidden_neurons(disc_values = c(-1L, 4L)), "positive integers")
    expect_error(hidden_neurons(disc_values = c(NA, 4L)), "positive integers")
})

test_that("activations, output_activation, optimizer, bias, bidirectional, validation_split, n_hlayers all construct", {
    expect_s3_class(activations(), "qual_param")
    expect_s3_class(output_activation(), "qual_param")
    expect_s3_class(optimizer(), "qual_param")
    expect_s3_class(bias(), "qual_param")
    expect_s3_class(bidirectional(), "qual_param")
    expect_s3_class(validation_split(), "quant_param")
    expect_s3_class(n_hlayers(), "quant_param")
})

# ---- grid_depth.R: dispatch methods ----

test_that("grid_depth.list dispatches through to grid_depth.parameters", {
    g = grid_depth(list(hidden_neurons(c(8L, 32L)), activations(c("relu", "elu"))),
                   n_hlayer = 1L, type = "regular", levels = 2L)
    expect_true(is.data.frame(g))
})

test_that("grid_depth.default errors for unsupported classes", {
    expect_error(grid_depth.default(1:3), "No method")
})

test_that("grid_depth.param combines multiple param objects via dots", {
    g = grid_depth(hidden_neurons(c(8L, 16L)), optimizer(), n_hlayer = 1L, type = "random", size = 3)
    expect_equal(nrow(g), 3)
})

# ---- generate_regular_grid: neurons + activations combined ----

test_that("generate_regular_grid builds an architecture grid with neurons and activations", {
    g = grid_depth(hidden_neurons(c(8L, 16L)), activations(c("relu", "elu")),
                   n_hlayer = 1L, type = "regular", levels = 2L)
    expect_true("hidden_neurons" %in% names(g))
    expect_true("activations" %in% names(g))
})

test_that("generate_regular_grid handles neurons-only and activations-only", {
    g1 = grid_depth(hidden_neurons(c(8L, 16L)), n_hlayer = 1L, type = "regular", levels = 2L)
    expect_true("hidden_neurons" %in% names(g1))

    g2 = grid_depth(activations(c("relu", "elu")), n_hlayer = 1L, type = "regular", levels = 2L)
    expect_true("activations" %in% names(g2))
})

test_that("generate_regular_grid folds in scalar params via crossing", {
    g = grid_depth(hidden_neurons(c(8L, 16L)), optimizer(), n_hlayer = 1L, type = "regular", levels = 2L)
    expect_true("optimizer" %in% names(g))
})

# ---- generate_random_grid ----

test_that("generate_random_grid samples depth from a vector of n_hlayer values", {
    g = grid_depth(hidden_neurons(c(8L, 32L)), activations(c("relu", "elu")),
                   n_hlayer = c(1L, 2L), type = "random", size = 6)
    expect_equal(nrow(g), 6)
})

# ---- generate_lhs_grid ----

test_that("generate_lhs_grid produces a design of the requested size", {
    g = grid_depth(hidden_neurons(c(8L, 64L)), optimizer(), n_hlayer = 1L, type = "latin_hypercube", size = 4)
    expect_equal(nrow(g), 4)
})

test_that("generate_lhs_grid falls back to random grid when there are no numeric dims", {
    g = grid_depth(activations(c("relu", "elu")), n_hlayer = 1L, type = "latin_hypercube", size = 3)
    expect_equal(nrow(g), 3)
})

# ---- extract_param_range branches ----

test_that("extract_param_range returns explicit values when param$values is set", {
    p = hidden_neurons(disc_values = c(4L, 8L, 16L))
    vals = extract_param_range(p, levels = 3L)
    expect_equal(sort(vals), c(4L, 8L, 16L))
})

test_that("extract_param_range with no levels returns full range for double/integer", {
    p_int = hidden_neurons(range = c(4L, 40L))
    expect_equal(extract_param_range(p_int, levels = NULL), 4:40)

    p_dbl = validation_split(range = c(0, 0.4))
    expect_equal(extract_param_range(p_dbl, levels = NULL), c(0, 0.4))
})

test_that("extract_param_range returns NULL for NULL param or character param", {
    expect_null(extract_param_range(NULL, levels = 3L))
})

# ---- sample_from_param ----

test_that("sample_from_param samples from explicit values and from a range", {
    p_vals = hidden_neurons(disc_values = c(4L, 8L))
    s1 = sample_from_param(p_vals, 5)
    expect_true(all(s1 %in% c(4L, 8L)))

    p_range = hidden_neurons(range = c(4L, 64L))
    s2 = sample_from_param(p_range, 5)
    expect_true(all(s2 >= 4L & s2 <= 64L))

    expect_null(sample_from_param(NULL, 5))
})

# ---- safe_sample and %||% ----

test_that("safe_sample repeats a length-1 vector rather than sampling it", {
    expect_equal(safe_sample(5, 3), c(5, 5, 5))
})

test_that("%||% returns y only when x is NULL", {
    expect_equal(1 %||% 2, 1)
    expect_equal(NULL %||% 2, 2)
})

# ---- count_numeric_params / decode_scalars ----

test_that("count_numeric_params counts only double/integer params", {
    params = list(optimizer(), validation_split())
    expect_equal(count_numeric_params(params), 1)
})

test_that("decode_scalars decodes numeric and categorical params, and handles the empty case", {
    params = list(vs = validation_split(range = c(0, 1)), opt = optimizer())
    out = decode_scalars(params, design_vals = c(0.5))
    expect_true(all(c("vs", "opt") %in% names(out)))

    expect_equal(nrow(decode_scalars(list())), 0)
})

# ---- table_summary.R: style and alignment paths ----

test_that("table_summary applies named cli styles to border and title", {
    df = data.frame(type = c("a", "b"), res = c("1", "2"))
    expect_output(table_summary(df, title = "My Table",
                                style = list(border_text = "bold", title = "blue")))
})

test_that("table_summary accepts a function-based style", {
    df = data.frame(type = c("a", "b"), res = c("1", "2"))
    expect_output(table_summary(df, style = list(
        left_col = function(x) toupper(x$value),
        right_col = "red"
    )))
})

test_that("align_test centers text within a given width", {
    out = align_test("hi", 6)
    expect_equal(nchar(out), 6)
})

test_that("format_row_summary supports left/right/center alignment", {
    r_left = format_row_summary("a", "b", 5, 5, align = "left")
    r_right = format_row_summary("a", "b", 5, 5, align = "right")
    r_center = format_row_summary("a", "b", 5, 5, align = "center")
    expect_true(is.character(r_left))
    expect_true(is.character(r_right))
    expect_true(is.character(r_center))
    expect_false(identical(r_left, r_right))
})

test_that("format_row_summary supports separate left_col/right_col alignment via list", {
    out = format_row_summary("a", "b", 5, 5, align = list(left_col = "right", right_col = "left"))
    expect_true(is.character(out))
})
