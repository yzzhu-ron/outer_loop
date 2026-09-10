# Semantic search-probe pilot

This continuation adds local semantic rewriting, hypothetical passages and
two-query decomposition to the original and keyword actions. It compares
random, fixed-coverage, Gaussian-information-gain and source-trained one-step
value selection on SciFact, FiQA and NFCorpus, with BM25 and MiniLM retrieval.

The [research report](../searchprobe-research-report.html) connects the measured
results, finite-model theory and [SearchProbe package](../../../packages/searchprobe/).
The original [lexical pilot](../pilot/RESULTS.md) is preserved.

All three corpora are exploratory families. Each leave-one-corpus-out fold uses
the other two corpora for source training, calibration and selector-utility
labels. Onboarding has no future target queries or relevance labels. The full
corpus remains indexed; eligible evaluation queries have all positive support
documents in a duplicate-group held-out pool. NFCorpus retains only 65 of 323
test queries, a substantial distribution limitation.

## Reproduce

The realized run uses Python 3.14 on an Apple Silicon Mac with 48 GiB RAM.
MLX generation requires Apple Silicon/Metal. Retrieval, analysis and package
tests use CPU; the package itself needs only Python 3.10 or later. No paid model
API is used. Creating an equivalent generator on other hardware is a new run,
not a claim of bitwise reproduction.

From the repository root:

```sh
uv venv --python 3.14 papers/learning-to-explore-search-environments/pilot/.venv
uv pip install --python papers/learning-to-explore-search-environments/pilot/.venv/bin/python \
  -r papers/learning-to-explore-search-environments/semantic_pilot/requirements.lock
```

The following shell variables shorten the commands without changing system
environment variables. The downloader verifies the three corpus archives and
downloads immutable MiniLM and quantized Qwen snapshots (about 2.2 GB of models).

```sh
pilot_python=papers/learning-to-explore-search-environments/pilot/.venv/bin/python
semantic_dir=papers/learning-to-explore-search-environments/semantic_pilot
"$pilot_python" "$semantic_dir/prepare_data.py"
TOKENIZERS_PARALLELISM=false "$pilot_python" "$semantic_dir/run_semantic.py" prepare
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  "$pilot_python" "$semantic_dir/run_semantic.py" retrieve
"$pilot_python" "$semantic_dir/run_semantic.py" analyze
MPLCONFIGDIR=/tmp/searchprobe-matplotlib "$pilot_python" "$semantic_dir/analyze_diagnostics.py"
"$pilot_python" "$semantic_dir/export_generations.py"
"$pilot_python" "$semantic_dir/audit_trace_exports.py"
"$pilot_python" "$semantic_dir/audit_router_sensitivity.py" --include-full-pool
"$pilot_python" "$semantic_dir/audit_source_decisions.py"
"$pilot_python" "$semantic_dir/audit_response_models.py"
"$pilot_python" "$semantic_dir/compare_lexical.py"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$pilot_python" "$semantic_dir/world_model_followup.py"
MPLCONFIGDIR=/tmp/searchprobe-matplotlib "$pilot_python" "$semantic_dir/plot_costs.py"
"$pilot_python" "$semantic_dir/build_report.py"
```

Use `prepare_data.py --verify-only` to check already-downloaded files without
network access. Intermediate generations and retrieval outcomes are cached.
The runner rejects generation/retrieval code-contract drift. Finish changes to
the runner, generator or main protocol before generation; if changing them is
necessary, re-run preparation and retrieval with the retained generation cache.
Generation cache keys include prompts, model metadata and generation settings.

## Inspect the evidence

- [protocol.json](protocol.json) and [analysis_protocol.json](analysis_protocol.json)
  specify the main experiment and descriptive comparisons before semantic
  retrieval results. Source hyperparameters are frozen; no target-label tuning.
- `results/headroom.csv` contains fixed-action and per-query oracle quality.
  `adaptation.csv`, `paired_outcomes.json`, `profiles.json` and `models.json`
  retain every method/seed/budget and the source-trained model metadata.
- `results/selector_diagnostics.json` reports the amount of nonzero one-step
  source utility signal. Repeated source histories are not independent labels.
- [world_model_protocol.json](world_model_protocol.json) fixes the exploratory
  responsiveness intervention after the original result and before its own
  evaluation, recorded in commit `64cab28`. `results/world_model_followup/`
  contains all four methods, 360 run rows, 1,440 cost rows, posterior traces,
  fitted source worlds and paired comparisons. Its primary reference is its
  own prior router, separating probing from a change of starting policy.
- `results/response_models/` exports the original router's scores and budget-32
  bounds as reusable SearchProbe inputs. The generic exact conditional audit
  reproduces the six label-free unchanged certificates. It assumes the supplied
  bounds cover the router; it does not formally verify floating-point code.
- `results/source_decision_models/` applies the SearchProbe decision audit to
  each fold's two source-training utility vectors. The saved models are usable
  CLI inputs. Their radius concerns the best fixed action in a declared finite
  source model; it is neither per-query headroom nor a target-regret guarantee.
- `results/fusion_baseline.csv` evaluates original plus a non-original action
  selected by source training utility. Fusion uses saved top-10 rankings; a
  decomposition action is already fused, so combining it is intentionally nested.
- `results/paired_intervals.csv` contains conditional query/seed bootstrap
  intervals. Three corpus families and three onboarding seeds do not establish
  population-level transfer or support post-selection significance claims.
- `results/generation_records.jsonl` preserves actual generated text, invalid
  responses, cost metadata and source IDs without relevance labels. The
  [fixed manual inspection](GENERATION_AUDIT.md) is an assistant-conducted sample
  audit, not an independent human or dataset-wide semantic validation.
- `results/analysis_metadata.json`, `manifest.json`,
  `diagnostics_provenance.json` and `generation_export.json` retain provenance.
  Trusted fitted model bundles remain in ignored `cache/fitted_models/`; their
  checksums and explicit base/scaler/calibrator parameters are saved in results.

`results/probe_traces/` contains experiment action-input renderings and nominal
per-selected-pair search costs. A decomposed action's newline rendering combines
two actual search requests; it is not a single submitted query. The export audit
produces strict single-query logs and reports under `results/strict_probe_traces/`
and `results/strict_probe_diagnostics/`, excluding decomposed pairs with disclosed
counts. Read `results/trace_provenance.json` before aggregating these logs.

## Costs and limitations

Each positive-budget method pays for inspecting its entire 16-document pool and
for all question and rewrite generations, including invalid or unselected
candidates. Selected probes cost two searches, or three for decomposition.
Budget zero pays no onboarding cost. Every generated serving action pays for
the full shared rewrite response; malformed outputs are retained as unavailable
actions with zero retrieval utility. Logical costs remain charged on cache hits.

Sampling calls, search calls, LLM calls, input/output tokens and measured setup
generation time are reported separately. Setup time is measured batch wall time
amortized over batch members; missing serving latency is not zero latency.
The frozen query encoder supplies router features; its shared encoding time is
in the retrieval manifest, but no per-task encoder price or latency is imputed.
Index construction, cache generation and source fitting are offline work. Equal
pair counts are not equal total costs; the cost curves retain the difference.

Known-document reciprocal rank is a synthetic proxy: other answers can be
relevant, and generated questions can change or overstate their source. The
256-token dense input limit truncates long documents and queries. HyDE uses
synthetic passages as retrieval representations, never evidence. The experiment
tests retrieval routing, not end-to-end answer correctness or agent performance.

## Checks

```sh
"$pilot_python" -m unittest discover -s "$semantic_dir/tests" -v
"$pilot_python" -m unittest discover -s papers/learning-to-explore-search-environments/pilot/tests -v
python3 -m unittest discover -s papers/learning-to-explore-search-environments/theory -v
PYTHONPATH=packages/searchprobe/src python3 -m unittest discover -s packages/searchprobe/tests -v
```

Tests cover query/document separation, unavailable-action costs, source/target
fitting boundaries, selected-outcome callbacks, matched budget prefixes,
cohort/provenance checks, paired uncertainty and the finite decision calculations.
