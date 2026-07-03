# JOSS pre-submission check — kindling package

Working document, not intended for submission to JOSS. Audit performed
on 2026-07-03 on the `joss-submission` branch, following the JOSS review
guidelines
(<https://joss.readthedocs.io/en/latest/review_criteria.html>). The
paper (`paper.md`/`paper.bib`) does not exist yet and is out of scope
here.

Audit environment: R 4.5.2 (aarch64-apple-darwin20), devtools, torch
(`mps` device detected locally), covr, roxygen2 7.3.3.

Status legend: **fixed automatically** / **needs Antoine/Joshua’s call**
/ **no action needed**.

------------------------------------------------------------------------

## 1. Installation & build

| Severity | Finding | Status |
|----|----|----|
| — | `devtools::install()`: clean install in ~15s, package loads and reports version `0.3.1.9000`. | No action needed |
| Cosmetic | [`library(kindling)`](https://kindling.joshuamarie.com) prints a masking message: `The following object is masked from 'package:base': args`. This is intentional (kindling exports an [`args()`](https://kindling.joshuamarie.com/dev/reference/args.md) helper marked *superseded*), but may surprise a new user. | Needs a call (keep as-is or rename) |
| — | `devtools::check(args = "--as-cran")`: first run → **1 ERROR** (failed rebuild of the `tuning-capabilities.Rmd` vignette, cause: the `Suggests` package `modeldata` was not installed in the audit environment) + 1 transient NOTE (`unable to verify current time`, a system-clock artifact, gone on the next run). | Fixed (installed `modeldata` in the audit environment, no code change needed — the dependency was correctly declared) |
| — | After installing `modeldata`: **`devtools::check()` → Status: OK, 0 error, 0 warning, 0 note** (reproduced twice, ~3-4 min each run). | No action needed — JOSS blocker cleared |

**Section 1 conclusion**: the package installs and passes
`R CMD check --as-cran` with no error, warning, or note. Nothing
blocking for JOSS/CRAN.

------------------------------------------------------------------------

## 2. Tests

| Severity | Finding | Status |
|----|----|----|
| — | `devtools::test()` (full suite, including torch): **205 `test_that()` blocks, 206 expectations, 0 failures, 0 errors surfaced**, 35 warnings — all caused by `tput: No value for $TERM and no -T specified` (an artifact of this audit’s sandboxed shell, which has no `$TERM` set; does not occur in the GitHub Actions workflows). Total runtime: **~33 seconds**, well under the 30-45 min threshold — no need for a background run. | No action needed |
| **Needs a fix** | **Reproducible bug found** (not fixed, touches code logic): in `tests/testthat/test-tune-workflows.R`, the very first test (“Tuning mlp_kindling with grid_depth works”) triggers a low-level torch error inside [`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html): `argument "weight" is missing, with no default`. The test doesn’t “see” the failure because `tune_grid()` catches errors per resampling fold, and the test only asserts `nrow(collect_metrics(out)) > 0` (true because the other fold happens to succeed). Full traceback obtained: `torch::nnf_linear(x)` → `torch_linear(input, weight, bias)` → missing `weight` argument. **Root cause identified**: in `R/act-fun.R`, `validate_activation()` only checks that `torch::nnf_<name>` exists. When `output_activation = "linear"`, this resolves to [`torch::nnf_linear`](https://torch.mlverse.org/docs/reference/nnf_linear.html) — which does exist in [torch](https://torch.mlverse.org/docs), but is the *functional* form of a linear layer (it explicitly requires `weight` and `bias` tensors), not an identity/no-op activation. The generated model then calls `torch::nnf_linear(x)` with a single argument → crash at forward-pass time. **This exact pattern, `output_activation(c("sigmoid", "linear"))`, is used as-is in the README’s “Hyperparameter Tuning” example** — so a user copy-pasting that example can get silently partial results or a crash depending on how the grid happens to be sampled. Note: the default values of [`output_activation()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md) (in `R/dials-kindling.R`) do **not** include `"linear"` or `"sigmoid"` — only users who customize the `values` vector (as the README does) are exposed. | **Needs Joshua’s call** — a logic/API decision: either forbid `"linear"`/map it to a proper identity function, or clearly document that it isn’t a valid output activation and remove it from the README example. Not fixed here — filed as [issue \#21](https://github.com/joshuamarie/kindling/issues/21). |
| Info | Spot-checked several test files (`test-tune-workflows.R`, `test-prints.R`, `test-act_funs.R`): substantive assertions (`expect_equal`, `expect_s3_class`, `expect_error` with condition classes) — no empty or “always true” tests found. Not an exhaustive check across all 23 files / 3557 lines. | No action needed |
| Info | [`covr::package_coverage()`](http://covr.r-lib.org/reference/package_coverage.md): overall coverage **68.16%**. Weakest files among exports: `R/table_summary.R` (54.17%), `R/early_stop.R` (54.55%), `R/grid_depth.R` (61.22%), `R/loss-nn-validator.R` (66.67%), `R/nn_arch.R` (68%), `R/act-fun.R` (70.88%). `R/register-parsnip.R` and `R/zzz.R` show 0% but this is a measurement artifact: this code runs in `.onLoad()` before covr’s instrumentation attaches, so it is in fact exercised indirectly by every parsnip integration test. `R/check-device.R` (20%) and `R/check-optimizer.R` (30.77%) have hardware-dependent branches (no CUDA on this runner) that are hard to cover without a multi-GPU CI matrix. | Needs a call (optional improvement, not a JOSS blocker — JOSS requires meaningful tests, not a coverage threshold) |

**Section 2 conclusion**: healthy, fast test suite, but a real logic bug
(`output_activation = "linear"`) was uncovered, and it’s documented in
the README itself. Worth addressing before submission since it affects
the credibility of a public example.

------------------------------------------------------------------------

## 3. Documentation & API

| Severity | Finding | Status |
|----|----|----|
| — | 36 exported symbols (`NAMESPACE`), all documented (Rd aliases present for each). `R CMD check` confirms: “checking for missing documentation entries… OK”, “checking for code/documentation mismatches… OK”, “checking Rd sections… OK”, “checking Rd contents… OK”. | No action needed |
| Cosmetic | Three exported, user-facing functions had no `@examples`: [`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md), [`args()`](https://kindling.joshuamarie.com/dev/reference/args.md), [`early_stop()`](https://kindling.joshuamarie.com/dev/reference/early_stop.md). | **Fixed automatically** — added working examples in `R/act-fun.R` and `R/early_stop.R`, regenerated docs (`devtools::document()`), and verified each example runs without error via [`tools::Rd2ex()`](https://rdrr.io/r/tools/Rd2HTML.html) + [`source()`](https://rdrr.io/r/base/source.html). |
| Cosmetic | Small cleanup: 2 lines of dead/commented-out code in `R/act-fun.R` (`# fn = attr(params, "act_fn")` / `# function(x_expr) call2(fn, x_expr)`), a leftover from an earlier refactor, redundant with the active code right below it. | **Fixed automatically** (removed) |
| — | Internal Rd pages with no `\value` (`ffnn_impl.Rd`, `rnn_impl.Rd`, `train_nn_impl.Rd`, `train_nn_impl_dataset.Rd`, `dot-train_nn_tab_impl.Rd`, `make_kindling.Rd`, `prepare_kindling_args.Rd`, `safe_sample.Rd`) all correspond to **non-exported** functions, marked `@keyword internal` or absent from `NAMESPACE` — exempt from the CRAN requirement. `reexports.Rd` (garson/olden) and `kindling.Rd` (package doc) don’t need one either. | No action needed |
| — | The `predict.*`/`print.*` S3 methods are documented (full `@param`/`@return`) but marked `\keyword{internal}` with no dedicated example — usage examples for [`predict()`](https://rdrr.io/r/stats/predict.html) already exist at the constructor level ([`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md), [`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md), [`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)). Consistent documentation choice, nothing to gain by changing it. | No action needed |
| — | `devtools::run_examples()` equivalent: all vignettes/examples run without error (“checking examples… OK”, “checking examples with –run-donttest… OK”). | No action needed |

**Section 3 conclusion**: documentation is complete and consistent; 3
missing examples added and verified.

------------------------------------------------------------------------

## 4. Vignettes

| Severity | Finding | Status |
|----|----|----|
| — | All 5 vignettes (`custom-act-fn.Rmd`, `kindling.Rmd`, `similar-packages.Rmd`, `special-cases.Rmd`, `tuning-capabilities.Rmd`) rebuild without error in `R CMD check` once `modeldata` is installed (a `Suggests` dependency correctly declared in `DESCRIPTION`, simply missing from the initial audit environment). | No action needed |

**Section 4 conclusion**: nothing to flag on the package side.

------------------------------------------------------------------------

## 5. README reproducibility

| Severity | Finding | Status |
|----|----|----|
| — | Installation (`install.packages`, [`pak::pak`](https://pak.r-lib.org/reference/pak.html)) and Level 1 ([`ffnn_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_gens.md)): reproduced successfully. | No action needed |
| — | Level 2 ([`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md) + [`predict()`](https://rdrr.io/r/stats/predict.html) on iris): reproduced successfully (tested with fewer epochs for audit speed; the training behavior itself showed no anomaly). | No action needed |
| **Needs a fix** | **Reproducibility bug found and fixed**: the “Level 3: Conventional tidymodels Integration” block used `box::use(..., mlbench[Ionosphere])`, which fails with `name "Ionosphere" not exported by "mlbench"`. Cause: unlike `modeldata` (which has `LazyData: true`), the `mlbench` package does not declare `LazyData` and does not export its datasets through `NAMESPACE` — they’re only accessible via `data(Ionosphere, package = "mlbench")`, as a comment already present in the code hinted (`# data(Ionosphere, package = "mlbench")`). | **Fixed automatically** in `README.Rmd` and `README.md` (this is an illustrative code block, not executed at knit time — so both files were edited directly without a full re-render). The fix was verified by running the corrected block end-to-end (training + `yardstick` metrics obtained successfully). |
| **Needs a fix** | The “Hyperparameter Tuning & Resampling” block (grid `output_activation(c("sigmoid", "linear"))`) reproduces functionally (verified with a reduced grid), **but shares the root cause of the `"linear"` bug described in section 2** — not fixed here, since it’s a logic/API decision. | Needs Joshua’s call (same item as section 2) |
| Cosmetic | `tput: No value for $TERM...` warnings visible during local testing, related to [cli](https://cli.r-lib.org)-based [`print()`](https://rdrr.io/r/base/print.html) methods — artifact of this sandboxed terminal without `$TERM`, absent from a normal terminal or CI. | No action needed |

**Section 5 conclusion**: one real reproducibility bug fixed (the
`Ionosphere` dataset); a second issue (the `"linear"` activation)
documented but left for Joshua to decide.

------------------------------------------------------------------------

## 6. Statement of need / positioning (`vignettes/similar-packages.Rmd`)

Summary of the current content (not modified):

- Compares [kindling](https://kindling.joshuamarie.com) to
  [brulee](https://github.com/tidymodels/brulee),
  [cito](https://citoverse.github.io/cito/), and
  [luz](https://mlverse.github.io/luz/), all built on
  [torch](https://torch.mlverse.org/docs).
- Comparison table across 12 dimensions: primary focus, design
  philosophy, supported architectures, code generation, tidymodels
  integration, formula syntax, per-layer activations, GPU support,
  explainability/xAI, statistical inference, custom loss functions, and
  target audience.
- Positions [kindling](https://kindling.joshuamarie.com) as the unique
  combination of: architectural flexibility (FFNN + RNN/LSTM/GRU),
  inspectable code generation, and full
  [tidymodels](https://tidymodels.tidymodels.org) integration
  (parsnip/tune/recipes/workflows) — as opposed to
  [brulee](https://github.com/tidymodels/brulee) (production models),
  [cito](https://citoverse.github.io/cito/) (inference/xAI), and
  [luz](https://mlverse.github.io/luz/) (generic training loop).
- The “Complementary Use” section explains that these packages are
  complementary, not mutually exclusive.

Observations to anticipate the “state of the field” section of the
future `paper.md` (not fixed, just flagged):

- The table does not mention plain
  [torch](https://torch.mlverse.org/docs) itself as an explicit
  comparison point (only mentioned in the intro paragraph).
- [mlr3](https://mlr3.mlr-org.com) is mentioned in the README as
  “planned” for kindling, but doesn’t appear in the comparison vignette
  — a cross-reference could be useful.
- [mlr3torch](https://mlr3torch.mlr-org.com/) and the R bindings for
  Keras/TensorFlow ([keras3](https://keras3.posit.co/),
  [tensorflow](https://github.com/rstudio/tensorflow)) are not
  mentioned; a JOSS reviewer might expect to see why they’re out of
  scope for the comparison.
- The factual claims about
  [brulee](https://github.com/tidymodels/brulee)/[cito](https://citoverse.github.io/cito/)/[luz](https://mlverse.github.io/luz/)
  features have not been independently re-verified against their current
  CRAN pages (out of scope for this technical audit).

**Needs Antoine’s call** when drafting the paper.

------------------------------------------------------------------------

## 7. Metadata & JOSS compliance

| Severity | Finding | Status |
|----|----|----|
| — | License: `MIT + file LICENSE`, `LICENSE` (CRAN template) and `LICENSE.md` (full MIT text) present and consistent — OSI-approved license. | No action needed |
| — | `DESCRIPTION` authors/roles: Joshua Marie (aut, cre) + Antoine Soetewey (aut, ORCID `0000-0001-8159-0804`) — consistent with `inst/CITATION` (same two authors). | No action needed |
| Info | `DESCRIPTION` had an **uncommitted change already in progress before this audit**: Antoine Soetewey’s email changing from `ant.soetewey@gmail.com` to `antoine.soetewey@uclouvain.be`. That change is not part of this audit and was left out of the fix branch/PR on purpose. | Needs Antoine’s call — confirm the final email before committing/submitting (unrelated to this PR). |
| — | `URL`/`BugReports`: `https://kindling.joshuamarie.com` (HTTP 200), `https://github.com/joshuamarie/kindling` (HTTP 200), `.../issues` (HTTP 200) — all valid. | No action needed |
| — | `inst/CITATION`: consistent with `DESCRIPTION` (uses [`packageDescription()`](https://rdrr.io/r/utils/packageDescription.html) dynamically for Package/Title/Version; same author list). | No action needed |
| — | `.github/CONTRIBUTING.md` and `.github/CODE_OF_CONDUCT.md` are present and actionable: detailed fork/branch/PR process, tidyverse style, testthat required for contributions; Contributor-Covenant-style code of conduct with a contact email for reporting (`joshua.marie.k@gmail.com`). | No action needed |
| Cosmetic | `.Rbuildignore` references `^CODE_OF_CONDUCT\.md$` (a pattern for a root-level file) while the actual file lives at `.github/CODE_OF_CONDUCT.md` (already covered by `^\.github$`) — harmless but stale/redundant pattern. | Not fixed (no impact, flagged for information) |
| — | `NEWS.md`: `# kindling (development version)` section at the top, consistent with the `0.3.1.9000` version in `DESCRIPTION`. Currently has a single generic item (“More visualization supports”). No entry for this audit’s fixes. | Needs a call — whether to add a NEWS.md bullet for this audit’s fixes (dead code, roxygen examples, README `Ionosphere` fix), optional. |

**Section 7 conclusion**: overall good JOSS compliance. The only
genuinely open item is confirming Antoine’s email in `DESCRIPTION` (an
already-in-progress change, unrelated to this audit).

------------------------------------------------------------------------

## 8. CI/CD

| Severity | Finding | Status |
|----|----|----|
| — | `R-CMD-check.yaml`, `pkgdown.yaml`, `test-coverage.yaml`: all **green** across the ~15 most recent runs on the default `main` branch, including the latest push on 2026-07-03 (merge of PR \#19). No recent red run. | No action needed |

------------------------------------------------------------------------

## 9. General code quality

| Severity | Finding | Status |
|----|----|----|
| Cosmetic | 2 lines of dead, commented-out code in `R/act-fun.R` (see section 3). | **Fixed automatically** |
| — | No `TODO`/`FIXME`/`XXX`/`HACK` markers found in `R/`, `tests/`, `vignettes/`, `inst/`. | No action needed |
| — | No [`browser()`](https://rdrr.io/r/base/browser.html) or forgotten debug calls. | No action needed |
| — | No hardcoded absolute file paths (`/Users/`, `/home/`, `C:\`) in code or tests. | No action needed |
| — | No sensitive secrets/credentials found. The only “secret”-looking references are the standard `${{ secrets.GITHUB_TOKEN }}` / `${{ secrets.CODECOV_TOKEN }}` usages in the GitHub Actions workflows — normal practice. | No action needed |
| **Needs a fix** | `output_activation = "linear"` logic bug (see sections 2 and 5) — the only substantive code issue found. | Needs Joshua’s call — [issue \#21](https://github.com/joshuamarie/kindling/issues/21) |

------------------------------------------------------------------------

## Files changed in this branch/PR

    .Rbuildignore      added the line ^JOSS_PRECHECK\.md$
    R/act-fun.R        fixed: removed dead code + added @examples (act_funs, args)
    R/early_stop.R     fixed: added @examples
    README.Rmd         fixed: box::use(mlbench[Ionosphere]) -> data(Ionosphere, package="mlbench")
    README.md          fixed: same (mirrors the non-executed block)
    man/act_funs.Rd    regenerated (roxygen2)
    man/args.Rd        regenerated (roxygen2)
    man/early_stop.Rd  regenerated (roxygen2)
    JOSS_PRECHECK.md   new — this report

Deliberately **not** included in this branch (unrelated, pre-existing
in-progress changes on Antoine’s local `joss-submission` branch, kept
separate on purpose):

    DESCRIPTION      — Antoine's email update, still under discussion
    .gitignore       — unrelated local additions (.quarto/ paths)
    man/kindling.Rd  — only changed as a side-effect of the DESCRIPTION email edit

------------------------------------------------------------------------

## Summary — ready to submit?

**Not yet** — but very close. The package is technically solid (clean
`R CMD check`, fast and non-trivial tests, green CI, complete
documentation, license/CoC/Contributing all in order). Three priority
actions remain, all light:

1.  **\[Priority 1 — logic\]** Decide the fate of
    `output_activation = "linear"` (and possibly `"sigmoid"`, pending
    the same check) which triggers a real torch crash
    (`argument "weight" is missing`) at training time. This exact
    pattern appears in the README’s public example (“Hyperparameter
    Tuning” section) — a JOSS reviewer running it could hit it. Filed as
    [issue \#21](https://github.com/joshuamarie/kindling/issues/21) for
    Joshua to decide: forbid these values, map them to a proper identity
    function, or at minimum remove the example from the README until
    fixed.
2.  **\[Priority 2 — metadata\]** Confirm Antoine Soetewey’s final email
    in `DESCRIPTION` (change already in progress, unrelated to this
    audit) and commit that choice.
3.  **\[Priority 3 — optional\]** Decide whether this audit’s fixes
    (dead code, roxygen examples, README fix) deserve a `NEWS.md` entry,
    and whether test coverage on
    `early_stop.R`/`grid_depth.R`/`table_summary.R` (54-61%) is worth
    strengthening before submission (not a JOSS blocker).

Once item 1 is settled (fixed or documented as a known limitation) and
item 2 is committed, the package is technically ready for JOSS
submission — all that’s left is writing `paper.md`/`paper.bib`, which is
out of scope for this check.
