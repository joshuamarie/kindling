#' Prepare arguments for kindling models
#' @keywords internal
prepare_kindling_args = function(args) {
    if ("hidden_neurons" %in% names(args) && is.list(args$hidden_neurons)) {
        if (length(args$hidden_neurons) == 1) {
            args$hidden_neurons = args$hidden_neurons[[1]]
        }
    }

    if ("activations" %in% names(args) && is.list(args$activations)) {
        if (length(args$activations) == 1) {
            args$activations = args$activations[[1]]
        }
    }

    args
}
