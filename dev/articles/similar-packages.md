# Similar packages and comparison

## Similar packages

All packages discussed here are built on top of
[torch](https://torch.mlverse.org/docs), R’s native implementation of
PyTorch. The torch package provides low-level tensor operations and
neural network building blocks, but requires substantial boilerplate
code for training. Higher-level packages like
[kindling](https://github.com/joshuamarie/kindling),
[brulee](https://github.com/tidymodels/brulee),
[cito](https://citoverse.github.io/cito/), and
[luz](https://mlverse.github.io/luz/) simplify this process while
offering different features and design philosophies.

[kindling](https://github.com/joshuamarie/kindling) distinguishes itself
through its unique code generation approach, versatile neural
architecture support (can be expanded more in the future), and
three-level API design. While
[brulee](https://github.com/tidymodels/brulee) focuses on
production-ready statistical models,
[cito](https://citoverse.github.io/cito/) emphasizes explainability and
statistical inference, and [luz](https://mlverse.github.io/luz/)
provides adaptable training loops.
[kindling](https://github.com/joshuamarie/kindling) is different but not
mutually exclusive to them: it offers deep architectural control and
bridges the gap between torch code and tidymodels workflows.

### Package Comparison

| Feature                        | kindling                                                                                                                     | brulee                                                       | cito                                                                | luz                                                                            |
|:-------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------|:--------------------------------------------------------------------|:-------------------------------------------------------------------------------|
| **Primary Focus**              | Architectural versatility & flexibility, statistical modelling, and code generation                                          | Production-ready statistical models                          | Statistical inference & interpretation                              | Training loop abstraction                                                      |
| **Design Philosophy**          | Three-level API (code gen, training, ML framework (currently tidymodels) integration)                                        | Batteries-included with sensible defaults                    | User-friendly with comprehensive xAI pipeline                       | High-level API reducing boilerplate                                            |
| **Architectures**              | Versatile — Feedforward Neural Networks (DNN/FFNN/MLP), Recurrent Neural Networks (RNN, LSTM, GRU), and more (in the future) | MLP, Linear/Logistic/Multinomial regression                  | Fully-connected networks, CNNs                                      | Any torch nn_module                                                            |
| **Code Generation**            | Yes (inspect & modify torch code)                                                                                            | No                                                           | No                                                                  | No                                                                             |
| **tidymodels Integration**     | Full (parsnip models & tuning)                                                                                               | Full (official tidymodels package)                           | No (standalone package)                                             | No (standalone package)                                                        |
| **Formula Syntax**             | Yes                                                                                                                          | Yes                                                          | Yes                                                                 | No (uses torch modules directly)                                               |
| **Layer-specific Activations** | Yes                                                                                                                          | No                                                           | No                                                                  | No (also uses torch modules directly)                                          |
| **GPU Support**                | Yes                                                                                                                          | Yes                                                          | Yes (CPU, GPU, MacOS)                                               | Yes (automatic device placement)                                               |
| **Explainability/xAI**         | Garson’s & Olden’s algorithms, vip integration, and more in the future                                                       | Limited                                                      | Extensive (PDP, ALE, variable importance, etc.)                     | No                                                                             |
| **Statistical Inference**      | Not yet implemented                                                                                                          | No                                                           | Yes (confidence intervals, p-values via bootstrap)                  | No                                                                             |
| **Custom Loss Functions**      | Yes                                                                                                                          | No                                                           | Yes                                                                 | Yes                                                                            |
| **For whom?**                  | Wanted versatile architectures (more in the future), fine-grained control, tidymodels users                                  | Wants standard supervised learning, stable production models | Do ecological modeling, interpretable models, statistical inference | Wants custom architectures, users needing human-friendly training loop control |

## Complementary Use

These packages aren’t mutually exclusive. You can use
[kindling](https://github.com/joshuamarie/kindling) and
[brulee](https://github.com/tidymodels/brulee) for production MLPs,
except [kindling](https://github.com/joshuamarie/kindling) provides
RNNs. The [cito](https://citoverse.github.io/cito/) package is when you
need oriented model interpretation. Then, the
[luz](https://mlverse.github.io/luz/) package is ideal when you want
less verbose training loops. Despite the difference between the
philosophies and main usage, all integrate with or build upon the torch
ecosystem, allowing you to switch between them as your modeling needs
evolve.

For instance, prototyping with
[kindling](https://github.com/joshuamarie/kindling) to explore different
network architectures is much easier, as well as deploying its models
for production, just like
[brulee](https://github.com/tidymodels/brulee), but
[cito](https://citoverse.github.io/cito/) is when stakeholders need
detailed explanations of model predictions.
