#' Variable Importance Methods for kindling Models
#'
#' This file implements methods for variable importance generics from
#' NeuralNetTools and vip packages.
#'
#' @name kindling-varimp
#'
#' @references
#'
#' Beck, M.W. 2018. NeuralNetTools: Visualization and Analysis Tools for Neural Networks. Journal of Statistical Software. 85(11):1-20.
#'
#' Garson, G.D. 1991. Interpreting neural network connection weights. Artificial Intelligence Expert. 6(4):46-51.
#'
#' Goh, A.T.C. 1995. Back-propagation neural networks for modeling complex systems. Artificial Intelligence in Engineering. 9(3):143-151.
#'
#' Olden, J.D., Jackson, D.A. 2002. Illuminating the 'black-box': a randomization approach for understanding variable contributions in artificial neural networks. Ecological Modelling. 154:135-150.
#'
#' Olden, J.D., Joy, M.K., Death, R.G. 2004. An accurate comparison of methods for quantifying variable importance in artificial neural networks using simulated data. Ecological Modelling. 178:389-397.
NULL

#' @importFrom NeuralNetTools garson
#' @export
NeuralNetTools::garson

#' @importFrom NeuralNetTools olden
#' @export
NeuralNetTools::olden

#' @importFrom vip vi_model
#' @export
vip::vi_model

#' Extract Weight Matrices from FFNN Model
#'
#' @param mod_in A fitted ffnn_fit object
#'
#' @return List with input_weights and output_weights matrices
#' @noRd
extract_ffnn_weights = function(mod_in) {
    if (!is.null(mod_in$cached_weights)) {
        return(mod_in$cached_weights)
    }

    model = mod_in$model
    n_hidden = length(mod_in$hidden_neurons)
    input_layer = model$fc1
    W_input = as.matrix(input_layer$weight$cpu())
    output_layer = model$out
    W_output = as.matrix(output_layer$weight$cpu())
    intermediate_weights = list()
    if (n_hidden > 1) {
        for (i in seq_len(n_hidden - 1)) {
            layer = model[[paste0("fc", i + 1)]]
            intermediate_weights[[i]] = as.matrix(layer$weight$cpu())
        }
    }

    list(
        input = W_input,
        output = W_output,
        intermediate = intermediate_weights
    )
}

#' @section Garson's Algorithm for FFNN Models:
#' `{kindling}` inherits `NeuralNetTools::garson` to extract the variable
#' importance from the fitted `ffnn()` model.
#'
#' @rdname kindling-varimp
#'
#' @param mod_in A fitted model object of class "ffnn_fit".
#' @param bar_plot Logical. Whether to plot variable importance (default FALSE).
#' @param ... Additional arguments passed to NeuralNetTools plotting.
#'
#' @return A data frame with variable importance scores.
#'
#' @examples
#' # Directly use `NeuralNetTools::garson`
#' model_mlp = ffnn(
#'     Species ~ .,
#'     data = iris,
#'     hidden_neurons = c(64, 32),
#'     activations = "relu",
#'     epochs = 100,
#'     verbose = FALSE,
#'     cache_weights = TRUE
#' )
#'
#' model_mlp |>
#'     garson()
#'
#' @export
#' @method garson ffnn_fit
garson.ffnn_fit = function(mod_in, bar_plot = FALSE, ...) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }

    weights = extract_ffnn_weights(mod_in)

    W_in = abs(t(weights$input))
    W_out = abs(weights$output)

    n_features = nrow(W_in)
    n_outputs = nrow(W_out)

    if (length(weights$intermediate) > 0) {
        W_propagated = W_in
        for (W_layer in weights$intermediate) {
            W_propagated = W_propagated %*% t(abs(W_layer))
        }
        W_combined = W_propagated
    } else {
        W_combined = W_in
    }

    importance = numeric(n_features)
    for (i in seq_len(n_features)) {
        total = 0
        for (j in seq_len(n_outputs)) {
            for (k in seq_len(ncol(W_combined))) {
                total = total + W_combined[i, k] * W_out[j, k]
            }
        }
        importance[i] = total
    }

    total_importance = sum(importance)
    if (total_importance > 0) {
        importance = (importance / total_importance) * 100
    }

    out_gar = data.frame(
        x_names = mod_in$feature_names,
        y_names = mod_in$response_name,
        rel_imp = importance,
        stringsAsFactors = FALSE
    )

    out_gar = out_gar[order(out_gar$rel_imp, decreasing = TRUE), ]
    rownames(out_gar) = NULL

    class(out_gar) = c("garson", "data.frame")

    if (bar_plot && requireNamespace("ggplot2", quietly = TRUE)) {
        p = ggplot2::ggplot(out_gar) +
            ggplot2::aes(x = reorder(x_names, rel_imp), y = rel_imp) +
            ggplot2::geom_col(fill = "steelblue") +
            ggplot2::coord_flip() +
            ggplot2::labs(
                x = "Features",
                y = "Relative Importance",
                title = "Variable Importance (Olden Method)"
            ) +
            ggplot2::theme_minimal()

        print(p)
    }

    out_gar
}

#' @section Olden's Algorithm for FFNN Models:
#' `{kindling}` inherits `NeuralNetTools::olden` to extract the variable
#' importance from the fitted `ffnn()` model.
#'
#' @rdname kindling-varimp
#'
#' @param mod_in A fitted model object of class "ffnn_fit".
#' @param bar_plot Logical. Whether to plot variable importance (default TRUE).
#' @param ... Additional arguments passed to NeuralNetTools plotting.
#'
#' @export
#' @method olden ffnn_fit
olden.ffnn_fit = function(mod_in, bar_plot = TRUE, ...) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }

    weights = extract_ffnn_weights(mod_in)

    W_in = t(weights$input)
    W_out = weights$output

    n_features = nrow(W_in)
    n_outputs = nrow(W_out)

    if (length(weights$intermediate) > 0) {
        W_propagated = W_in
        for (W_layer in weights$intermediate) {
            W_propagated = W_propagated %*% t(W_layer)
        }
        W_combined = W_propagated
    } else {
        W_combined = W_in
    }

    importance_matrix = matrix(0, nrow = n_features, ncol = n_outputs)

    for (i in seq_len(n_features)) {
        for (o in seq_len(n_outputs)) {
            importance_matrix[i, o] = sum(W_combined[i, ] * W_out[o, ])
        }
    }

    importance = if (n_outputs > 1) {
        rowMeans(importance_matrix)
    } else {
        importance_matrix[, 1]
    }

    out_old = data.frame(
        x_names = mod_in$feature_names,
        y_names = mod_in$response_name,
        rel_imp = importance,
        stringsAsFactors = FALSE
    )

    out_old = out_old[order(abs(out_old$rel_imp), decreasing = TRUE), ]
    rownames(out_old) = NULL

    class(out_old) = c("olden", "data.frame")

    if (bar_plot && requireNamespace("ggplot2", quietly = TRUE)) {
        p = ggplot2::ggplot(out_old) +
            ggplot2::aes(x = reorder(x_names, rel_imp), y = rel_imp) +
            ggplot2::geom_col(fill = "steelblue") +
            ggplot2::coord_flip() +
            ggplot2::labs(
                x = "Features",
                y = "Relative Importance",
                title = "Variable Importance (Olden Method)"
            ) +
            ggplot2::theme_minimal()

        print(p)
    }

    out_old
}

#' @section Variable Importance via `{vip}` Package:
#' You can directly use `vip::vi()` and `vip::vi_model()` to extract the variable
#' importance from the fitted `ffnn()` model.
#'
#' @rdname kindling-varimp
#'
#' @param object A fitted model object of class "ffnn_fit".
#' @param type Type of algorithm to extract the variable importance.
#'  This must be one of the strings:
#'  - 'olden'
#'  - 'garson'
#' @param ... Additional arguments passed to methods.
#'
#' @return A tibble with columns "Variable" and "Importance"
#'    (via `vip::vi()` / `vip::vi_model()` only).
#'
#' @examples
#' # kindling also supports `vip::vi()` / `vip::vi_model()`
#' model_mlp |>
#'     vip::vi(type = 'garson') |>
#'     vip::vip()
#'
#' @export
#' @method vi_model ffnn_fit
vi_model.ffnn_fit = function(object, type = c("olden", "garson"),  ...) {
    type = match.arg(type)
    result = switch(
        type,
        olden = olden(object, bar_plot = FALSE, ...),
        garson = garson(object, bar_plot = FALSE, ...)
    )

    if (requireNamespace("tibble", quietly = TRUE)) {
        tibble::tibble(
            Variable = result$x_names,
            Importance = abs(result$rel_imp)
        )
    } else {
        data.frame(
            Variable = result$x_names,
            Importance = abs(result$rel_imp),
            stringsAsFactors = FALSE
        )
    }
}
