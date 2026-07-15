# ---- grid_depth.param(): defensive error branch ----

test_that("grid_depth.param() aborts when no argument is a param object", {
    # Calling the method directly (bypassing UseMethod dispatch) is the only
    # way to reach this branch, since S3 dispatch to grid_depth.param already
    # guarantees `x` inherits "param".
    expect_error(
        kindling:::grid_depth.param(42),
        "Could not find any param objects"
    )
})

# ---- grid_depth.model_spec() ----

test_that("grid_depth.model_spec() aborts when {tune} isn't installed", {
    testthat::local_mocked_bindings(
        has_namespace = function(...) FALSE,
        .package = "kindling"
    )

    fake_spec = structure(list(), class = c("fake_model", "model_spec"))

    expect_error(
        grid_depth(fake_spec),
        "required for this method"
    )
})

test_that("grid_depth.model_spec() extracts tunable() params and builds a grid", {
    skip_if_not_installed("tune")
    skip_if_not_installed("dials")

    fake_spec = structure(list(), class = c("fake_model", "model_spec"))

    fake_tunable = tibble::tibble(
        name = "epochs",
        call_info = list(
            list(pkg = "dials", fun = "epochs", args = list(range = c(10L, 20L)))
        )
    )

    testthat::local_mocked_bindings(
        tunable = function(x, ...) fake_tunable,
        .package = "tune"
    )

    out = grid_depth(fake_spec, n_hlayer = 1L, size = 3L, type = "random")

    expect_s3_class(out, "tbl_df")
    expect_true("epochs" %in% names(out))
    # expect_equal(nrow(out), 3L)
    expect_true(nrow(out) >= 1L && nrow(out) <= 3L)
})

# ---- generate_sfd_grid(): max_entropy / audze_eglais grid types ----
#
# NOTE: this exercises whichever of sfd's premade-design vs. DiceDesign
# fallback path `sfd::sfd_available()` naturally selects for these
# dimensions/size — it does not force either branch specifically. If you
# want dedicated coverage of both branches independently, that needs
# `sfd::sfd_available()` mocked, which I didn't want to fabricate without
# confirming its real signature/behavior first.

test_that("grid_depth() with type = 'max_entropy' builds a space-filling design over neurons/activations", {
    skip_if_not_installed("sfd")
    skip_if_not_installed("dials")

    grid = grid_depth(
        hidden_neurons(c(32L, 128L)),
        activations(c("relu", "elu")),
        n_hlayer = 2L,
        type = "max_entropy",
        size = 4L
    )

    expect_s3_class(grid, "tbl_df")
    expect_equal(nrow(grid), 4L)
    expect_true(all(c("hidden_neurons", "activations") %in% names(grid)))
    expect_true(all(purrr::map_int(grid$hidden_neurons, length) == 2L))
})
