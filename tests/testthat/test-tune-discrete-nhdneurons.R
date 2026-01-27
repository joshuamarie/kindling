skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("hidden_neurons() with discrete values works in tuning workflow", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("workflows")
    skip_if_not_installed("tune")
    skip_if_not_installed("rsample")
    skip_if_not_installed("recipes")
    skip_if_no_torch()
    
    mlp_spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = tune::tune(),
        activations = tune::tune(),
        epochs = 10
    )
    
    iris_recipe = recipes::recipe(Species ~ ., data = iris)
    wf = workflows::workflow() |>
        workflows::add_recipe(iris_recipe) |>
        workflows::add_model(mlp_spec)
    
    set.seed(123)
    folds = rsample::vfold_cv(iris, v = 2)
    
    grid = grid_depth(
        hidden_neurons(values = c(32, 64, 128)),
        activations(c("relu", "elu")),
        n_hlayer = 2,
        size = 6,
        type = "random"
    )
    
    expect_equal(nrow(grid), 6)
    all_neurons = unlist(grid$hidden_neurons)
    expect_true(all(all_neurons %in% c(32, 64, 128)))
    
    results = tune::tune_grid(
        wf,
        resamples = folds,
        grid = grid,
        control = tune::control_grid(save_pred = FALSE)
    )
    
    expect_s3_class(results, "tune_results")
    expect_true(nrow(tune::collect_metrics(results)) > 0)
    
    best = tune::select_best(results, metric = "accuracy")
    expect_s3_class(best, "tbl_df")
    expect_true("hidden_neurons" %in% names(best))
    
    best_neurons = unlist(best$hidden_neurons)
    expect_true(all(best_neurons %in% c(32, 64, 128)))
    
    final_wf = tune::finalize_workflow(wf, best)
    expect_s3_class(final_wf, "workflow")
    
    final_fit = parsnip::fit(final_wf, data = iris)
    expect_s3_class(final_fit, "workflow")
})
