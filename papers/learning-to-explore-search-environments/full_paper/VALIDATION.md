# Validation record

The completed research cycle is commit `99706427fd7d97af0b267d50aca9e1f1815e565d`.
[Its GitHub Actions run](https://github.com/yzzhu-ron/outer_loop/actions/runs/34487618134)
passed all four jobs: Python 3.10, 3.12, 3.14, and the development-analysis checks.
That run covers 135 distinct unit tests, plus article validation and installed-command
smoke checks. The Python matrix repeats the dependency-free tests across versions.
This record does not claim that later changes have passed that earlier run.

## A new user's installation

The exact GitHub installation pinned to `9970642` was tested in a fresh Python 3.13
environment outside the checkout. The installed package exported its three
synthetic examples and ran all diagnostics. The separate evidence replay verified
810 saved run summaries, 18 follow-up baselines, 600 paired-contrast means, and
12 conditional certificates from the 34 committed evidence files. NumPy,
scikit-learn, PyTorch, and MLX were absent. This confirms the portable analysis
path; it does not rerun retrieval, fitting, selection, or bootstrap intervals.
The actual evidence-download path was also checked against GitHub.

## Article and figures

- The article validator checks local links, unique IDs, primary displayed
  retrieval numbers, and template/builder/figure provenance. It found no external
  page assets. External reference links are not required to read the article or
  operate the example.
- Fifty interactive algorithm outputs, across five workload weights and two
  budgets, matched the independent exact controlled calculation within `1e-12`.
- All three initial figure buttons opened decoded SVG images. Close and Escape cleared
  the dialogs without duplicate IDs.
- Screenshots were inspected at desktop and mobile widths, in light and dark
  mode. There was no page overflow at 1440, 390, or 320 pixels. One browser-tool
  timeout coincided with an unexpected blank-page navigation; a fresh navigation
  passed the affected mobile check. No application defect was established.
- The three initial scientific figures have a separate manifest of input/output hashes,
  plotted values, and accessible descriptions. Capacity brackets are labeled as
  evaluator bounds, not confidence intervals.

## Scientific scope

All three natural collection families were previously inspected development data.
The controlled mechanisms, natural results, capacity bounds, and source-selected
baselines have separate protocols and explicit limits. Passing software checks
does not establish a publication-ready result or a valid transfer assumption.
The new intervention study has its own pre-outcome freeze and validation record;
its results are not covered by the earlier CI commit above.

## Intervention study and posthoc audits

The implementation and protocol were committed at `16f668d` before generating
new outcomes. Permitted inputs and 1,638 target policy decisions were committed
at `b6b55ef` before computing the new target test utilities. The frozen synthetic
suite passed 28 tests covering source/target boundaries, calibration, controls,
costs, and paired uncertainty.

An independent audit reconstructed decisions, resource counts, response errors,
and the advancement gate without importing the experiment's arithmetic. It also
verified all eight frozen implementation hashes and thirteen source hashes.
Independent endpoint scoring matched 40,560 nDCG/recall comparisons and 8,450
archived single-action scores to at most `3.33e-16`. Detailed records are in
[intervention_transfer/RESULTS.md](intervention_transfer/RESULTS.md).

Portable intervention replay ran in a fresh copy with no raw retrieval caches.
Every regenerated decision field matched except its new creation timestamp.
Evaluation used the unchanged original decision lock with its hash-bound labels
and reproduced byte-identical result JSON. The replay neither rewrote timestamps
nor rebound the archived labels. A helper fix makes explicit relative output
directories work when subprocesses change their working directory. Three new
helper tests passed, bringing the local intervention suite to 31 tests. An actual
replay using an explicit relative destination also passed.

The posthoc [capacity and source-support audit](theory/POSTHOC_INTERVENTION_REVIEW.md)
has reproducible scripts and hash-linked outputs. Both outputs reproduced exactly;
analytic examples check query weighting and nested policy-class ceilings.
The capacity calculation uses sealed test labels only as an evaluator. The
source-support calculation indexes source train/calibration utilities and query
metadata, with no target test scores entering its arithmetic. Neither audit
is a new registered experiment or a deployed policy.

The fourth scientific figure has its own input/output and generator hashes.
The builder checks both manifests and preserves the three original figures.
The expanded article validator checks all six displayed posthoc capacity values
in addition to the retrieval metrics. It reports 642 unique IDs, 225 links, and
zero external page assets.

Focused browser checks covered the new intervention section, its expandable
three-row capacity table, and the fourth inline figure at 1440, 390, and 320
pixels in light and dark mode, with no horizontal page overflow. Physical
enlargement clicks, decoded images, Close/Escape, and dialog cleanup passed on
desktop light/dark and 390-pixel light layouts. The 390-pixel dark image also
loaded, but the subsequent Escape cleanup wait timed out and the browser later
reported `about:blank`. Fresh navigation passed that layout's section/table
checks. This remains a browser-harness limitation, with no established article
defect. The unchanged interactive calculation was not redundantly retested.

The updated CI workflow includes 166 distinct unit tests, portable intervention
replay, and byte comparisons for both posthoc audit outputs on Linux. Its results
must be read from the run for the corresponding commit; the earlier `9970642`
run does not validate these additions.
