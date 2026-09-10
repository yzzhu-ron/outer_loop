# Query-specific utility transfer diagnostic

This experiment asks whether query-only source models can recover useful action-selection gains before more probe-selection work. It uses the existing semantic cache: three previously inspected development families, two backends, five actions, and 392 query features. It makes no new retrieval or generation calls and has no onboarding.

The [protocol](protocol.v1.json), code, tests, and portable [evidence](evidence.json) were [frozen](freeze.v1.json) at **2026-09-10 13:56:23 UTC**, before the registered fitting/evaluation. Five tests passed. The export is 7.27 MB and deduplicates query features across backends; raw queries, documents, retrieval indexes, and model downloads are unnecessary for replay.

For each held-out family/backend, training uses 128 queries from each of the other two same-backend source families. Calibration uses 32 separate queries per source family. The 15 candidates are a global action mean, standardized ridge at two regularization settings, and twelve cosine kNN variants with two neighborhood-balance schemes and optional global-mean shrinkage. The source-calibration winner is deployed without refitting.

Calibration is **within the source corpora**, not a new-family validation split. All target families were already inspected in preceding research, so these results are exploratory.

Read [RESULTS.md](RESULTS.md). Outputs in [results.v1](results.v1/summary.json) include:

- `primary_results.json`: the ten main methods/oracles in every environment.
- `all_results.json`: all 15 frozen candidates as well as main comparators.
- `selections.json` and `source_calibration_grid.json`: source-only model choices and their calibration evidence.
- `paired.json`: query-level choices, nDCG, and recall.
- `conditional_intervals.json`: descriptive paired-query intervals per reused environment.
- `reconstruction_audit.json`: exact agreement with archived existing-router decisions.

Reproduce with Python and NumPy/scikit-learn/SciPy (tested versions in `results.v1/summary.json`):

```sh
python3 -m unittest -v
python3 experiment.py --freeze
python3 experiment.py --run
```

The committed minimized evidence supports replay directly. `export_evidence.py` rebuilds it only when the original local semantic and natural-result caches are available. Hash drift is rejected; changed designs require a new version. Current fitting uses one numerical-library thread for this small problem.

Serving costs include the selected action's searches and shared-format generation usage. Query-aware models additionally need one MiniLM encoding; fixed and original methods do not. The recorded encoding time is an average from the original batched run, not a new online-latency measurement. RRF-all costs six searches and one shared generation, and is explicitly a stronger, more expensive inference-time comparator.
