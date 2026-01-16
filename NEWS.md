# kindling 0.1.1

## Bug fixes / CRAN compliance

* All tests that require torch/LibTorch are now properly skipped on CRAN
* Removed or skipped tests that assumed torch is not loaded
* Fixed title case in DESCRIPTION
* Corrected/removed invalid URLs

# kindling 0.1.2

## Resubmission

This is a resubmission.

In this version I have:

* Bumped version to 0.1.2
* Fixed title case in DESCRIPTION
* Removed invalid/unreachable URLs
* Wrapped all examples that require torch in `if (torch::torch_is_installed()) { … }`

# kindling 0.1.0

* Initial CRAN submission.
