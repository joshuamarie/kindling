#' @keywords internal
.onLoad = function(libname, pkgname) {
    vctrs::s3_register("vip::vi_model", "ffnn_fit")

    has_tidymodels_engines =
        requireNamespace("parsnip", quietly = TRUE) &&
        utils::packageVersion("parsnip") >= "1.0.0" &&
        requireNamespace("dials", quietly = TRUE) &&
        requireNamespace("tune", quietly = TRUE)

    if (has_tidymodels_engines) {
        make_kindling()
    }
}
