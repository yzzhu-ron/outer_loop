# Finite-channel retrieval development

This continuation replaces the earlier binary Gaussian source model with four categorical worlds: two source corpora crossed with BM25/dense. Backend identity is hidden from all policies. The source model fits only source training utilities and source probe observations. All six test configurations come from previously inspected development families.

Code, protocol, tests, and the minimized input dataset were committed in `b827076` before this new evaluation. [protocol.json](protocol.json) and [freeze.json](freeze.json) record the assumptions and hashes. This freeze does not make already-inspected data confirmatory.

The primary two-step rule at a maximum 64 onboarding search calls scores .38239 macro nDCG@10, versus .37819 random and .38051 information gain. Its own prior is .38423; original-query retrieval is .39571; RRF-all is .42118. The declared development gate fails. [results/](results/) retains 1,620 run rows,60 baseline rows, all per-query outcomes and selected-probe traces,3,780 paired comparison intervals, source models, and provenance. Intervals are descriptive paired query/seed resampling within an environment; no population-level significance is inferred from three families.

The [independent interpretation review](../theory/NATURAL_REVIEW.md) explains two-step finite-pool optimism, division by first-probe cost, update-only likelihood tempering, source-family dependence, and missing short-probe coverage. The heuristic neither implements exact full-pool planning nor produces calibrated target confidence.

## Reanalyze without generation caches

From the repository root, in a Python environment with NumPy:

```sh
python -m pip install numpy
python papers/learning-to-explore-search-environments/full_paper/natural/experiment.py \
  --output-dir /tmp/searchprobe-natural-replay
```

Use a new output directory: existing results are not overwritten. The run verifies all frozen hashes. It takes about 25 seconds in the recorded environment (Python 3.14, NumPy 2.5.3), including 2,000 bootstrap draws per comparison. Floating-point library changes may affect boundary ties and interval endpoints; output metadata records versions. This replays acquisition and statistics on saved retrieval outcomes; it does not regenerate those outcomes.

`export_evidence.py` describes how the 1.6MB input was extracted from the old semantic caches. Export is unnecessary for replay. It deliberately omits query text/features and retains only the source/test fields needed by this experiment, rankings/qrels for its fusion baseline, and observable probe metadata plus evaluator-owned outcomes.

## Post-hoc capacity diagnostic

After seeing the failed gate, [capacity_audit.py](capacity_audit.py) enumerates four-bucket action policies and checks whether any shared posterior can induce them. This evaluator may use target labels; the deployed policy may not. The saved [capacity_audit.json](capacity_audit.json) reports a verified deterministic-argmax witness and a weak-tie optimistic upper bound. Their difference is retained, notably in SciFact/BM25. Solver failures other than reported infeasibility stop the audit.

```sh
python -m pip install scipy
python papers/learning-to-explore-search-environments/full_paper/natural/capacity_audit.py
python -m unittest discover \
  -s papers/learning-to-explore-search-environments/full_paper/natural -p 'test_*.py' -v
```

The restriction is mainly the coarse query partition and five-action menu: macro target bucket oracle .408915, versus the shared-posterior relaxed bound .407657. Both are below RRF-all .421179, which combines actions and lies outside this menu. These diagnostics motivate a richer utility model; they do not prove onboarding useless for other models.
