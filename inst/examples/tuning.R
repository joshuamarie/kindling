box::use(
    kindling[
        mlp_kindling, hidden_neurons, activations, output_activation, grid_depth
    ],
    parsnip[fit, augment],
    recipes[recipe],
    workflows[workflow, add_recipe, add_model],
    rsample[vfold_cv],
    tune[tune_grid, tune, select_best, finalize_workflow, collect_metrics],
    dials[grid_random],
    yardstick[accuracy, roc_auc, metric_set, metrics]
)

mlp_tune_spec = mlp_kindling(
    mode = "classification",
    hidden_neurons = tune(),
    activations = tune(),
    output_activation = tune()
)

iris_folds = vfold_cv(iris, v = 3)
nn_wf = workflow() |>
    add_recipe(recipe(Species ~ ., data = iris)) |>
    add_model(mlp_tune_spec)

nn_grid = grid_random(
    hidden_neurons(c(32L, 128L)),
    activations(c("relu", "elu")),
    output_activation(c("sigmoid", "linear")),
    size = 10
)

nn_grid_depth = grid_depth(
    hidden_neurons(c(32L, 128L)),
    activations(c("relu", "elu")),
    output_activation(c("sigmoid", "linear")),
    n_hlayer = 2,
    size = 10,
    type = "latin_hypercube"
)

nn_tunes = tune::tune_grid(
    nn_wf,
    iris_folds,
    grid = nn_grid_depth
    # metrics = metric_set(accuracy, roc_auc)
)

best_nn = select_best(nn_tunes)
final_nn = finalize_workflow(nn_wf, best_nn)

# collect_metrics(final_nn)
final_nn_model = fit(final_nn, data = iris)
final_nn_model

final_nn_model |>
    augment(new_data = iris) |>
    metrics(truth = Species, estimate = .pred_class)
