#' Regularize Loss Function
#'
#' @param model Torch module
#' @param penalty Numeric. Regularization strength
#' @param penalty_type Character. Type of regularization
#' @param mixture Numeric. Elastic net mixing parameter
#'
#' @return Torch tensor with regularization loss
#' @noRd
regularizer = function(model, penalty, mixture = 0.5) {
    # Default: When regularization wasn't applied
    # Early return
    if (penalty == 0) {
        return(torch::torch_tensor(
            0,
            device = model$parameters[[1]]$device,
            dtype = torch::torch_float32()
        ))
    }
    
    reg_loss = torch::torch_tensor(
        0,
        device = model$parameters[[1]]$device,
        dtype = torch::torch_float32()
    )
    
    for (param in model$parameters) {
        if (length(param$size()) <= 1) next 
        
        if (mixture == 1) {
            reg_loss = reg_loss + torch::torch_sum(torch::torch_abs(param))
        } else if (mixture == 0) {
            reg_loss = reg_loss + 0.5 * torch::torch_sum(param$pow(2))
        } else {
            l1 = torch::torch_sum(torch::torch_abs(param))
            l2 = 0.5 * torch::torch_sum(param$pow(2))
            reg_loss = reg_loss + mixture * l1 + (1 - mixture) * l2
        }
    }
    
    penalty * reg_loss
}

#' Validate Regularization Parameters
#'
#' @param penalty Numeric
#' @param penalty_type Character
#' @param mixture Numeric
#'
#' @noRd
validate_regularization = function(penalty, mixture) {
    if (penalty < 0) {
        cli::cli_abort(c(
            "{.arg penalty} must be non-negative.",
            x = "You provided {.val {penalty}}."
        ))
    }
    
    if (mixture < 0 || mixture > 1) {
        cli::cli_abort(c(
            "{.arg mixture} must be between 0 and 1.",
            x = "You provided {.val {mixture}}."
        ))
    }
    
    invisible(NULL)
}
