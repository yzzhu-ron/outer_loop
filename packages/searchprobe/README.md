# searchprobe

Inspect search probes, decision models, and router responsiveness before spending a larger retrieval budget. `searchprobe` runs locally with Python's standard library and accepts logs from any search API.

| Command | Input | What it checks |
| --- | --- | --- |
| `audit` | Paired-query JSONL logs | Collisions, coverage, rank ties, saturation and declared costs |
| `decision-audit` | Explicit source-world utility matrix | Exact conditional minimax regret and a worst-case witness |
| `response-audit` | Base scores and correction bounds | Which decisions cannot change, with exact tie handling |

The tool grew out of a retrieval pilot in which many nominally different query actions produced identical strings. It separates this structural problem from a measured tie between different queries. Neither a clean audit nor a nonzero probe contrast establishes that a probe predicts useful behavior on future tasks.

An optional, separate [`decision-audit`](DECISION_AUDIT.md) command accepts a declared **source utility matrix** and computes its exact conditional minimax regret, action mixture, and worst-case witness. It requires source provenance and modeling assumptions. It does not infer target utilities from a query log or produce a target-regret guarantee.

[`response-audit`](RESPONSE_AUDIT.md) checks whether declared score-correction bounds can change a router's choices. It reports exact conditional stability and an upper bound on absolute mean utility change for utilities in `[0,1]`. The caller must justify the bounds; the command reads no labels and provides no target-regret or generalization guarantee.

## Try the installed examples

The development branch includes editable examples in the wheel. With Python 3.10 or later and Git, install and run them without cloning the repository:

```sh
python -m pip install "searchprobe @ git+https://github.com/yzzhu-ron/outer_loop.git@codex/searchprobe-full-paper#subdirectory=packages/searchprobe"
searchprobe-demo --output-dir searchprobe-demo
```

The demo makes no network or model calls. It writes three synthetic inputs, three computed reports, and a README with commands for editing and rerunning each input. Expect 5 valid probe pairs with 1 exact query collision, a source decision radius of `1/3`, and `1/3` of response rows certified unchanged. These are demonstrations, not research measurements. Use `searchprobe-demo response --output-dir response-example` to try only one command. Existing files are never overwritten; choose a new output directory when rerunning the demo.

For a replay of **recorded research outcomes**, use the [portable CPU replay](../../papers/learning-to-explore-search-environments/full_paper/replay/README.md). It recalculates statistics and certificates from 34 committed files, without the ignored experiment caches, MLX, or a GPU. It does not regenerate retrieval, training, or bootstrap intervals.

## Run an audit

From the repository root, install into your own virtual environment:

```sh
python -m pip install ./packages/searchprobe
searchprobe audit probes.jsonl --output report.json
```

No package has been published to PyPI. To run directly from a checkout without installation:

```sh
PYTHONPATH=packages/searchprobe/src python3 -m searchprobe audit \
  packages/searchprobe/examples/synthetic.jsonl --output /tmp/searchprobe-report.json
```

For an installation pinned to the earlier milestone's three diagnostic commands, without a
manual checkout (requires Git and access to this repository; this older commit predates `searchprobe-demo`):

```sh
python -m pip install "searchprobe @ git+https://github.com/yzzhu-ron/outer_loop.git@17c2680ca05f8abf9d6d11f354e20c20f28e576b#subdirectory=packages/searchprobe"
```

`searchprobe` is also a member of this repository's uv workspace. With uv, use `uv run --package searchprobe searchprobe audit probes.jsonl --output report.json` from the repository root.

The bundled example is **entirely synthetic**, including document IDs, outcomes, and costs. It demonstrates exact and token-bag collisions, a useful observed contrast, a floor tie, and a probe awaiting retrieval. It is not research evidence.

The CLI writes the full JSON report to `--output` and a short readable diagnosis to stderr. Omit `--output` to send JSON to stdout. Exit codes are `0` for a completed valid audit, `1` for warnings when `--fail-on-warnings` is set, and `2` for invalid records or invocation/file errors. A report is still written when individual input lines fail validation. An empty or wholly invalid log returns `2`.

## JSONL contract, version 1

Each nonblank UTF-8 line describes one paired trial. Record all queries as actually submitted to the search service. Use a unique ID for each trial, including repeated trials. Keep one environment and one endpoint definition per log; each pair should compare actions under the same search-service configuration. The audit cannot verify those conditions.

```json
{
  "probe_id": "tax-paraphrase-01",
  "family": "indirect-question",
  "actions": ["original", "rewrite"],
  "queries": ["what do I owe the government", "calculate income tax liability"],
  "known_target": {"id": "document-17", "ranks": [null, 4], "cutoff": 10},
  "cost": {"values": [1, 1], "total": 2, "unit": "search_calls"},
  "bucket": "indirect"
}
```

The expanded example above is for readability; put each full object on one line in the actual file.

| Field | Contract |
| --- | --- |
| `probe_id`, `family` | Required nonempty strings. IDs must be unique within an audit. |
| `actions` | Required pair of distinct, nonempty action names. |
| `queries` | Required pair of nonempty strings. Keep the same left/right order as `actions`, costs, and outcomes. |
| `cost` | Required object: two finite nonnegative `values`, their `total`, and a nonempty `unit`. The total must match the pair sum within relative tolerance `1e-9` or absolute tolerance `1e-12`. |
| `known_target` | Optional object with two `ranks` and a positive integer `cutoff`; `id` is optional. Ranks are one-based integers at most `cutoff`, or `null` for absence at that cutoff. Both sides refer to the same known target. |
| `outcomes` | Alternative to `known_target`: an object with two finite numeric `values`, optional `[minimum, maximum]` `bounds`, and optional `higher_is_better` boolean (default `true`). Values must lie in declared bounds. |
| `bucket` | Optional nonempty feature-bucket string. Missing buckets remain a separate `null` coverage cell. |

Supply `known_target`, `outcomes`, or neither. **Omitting both is preflight mode**, which checks query structure, coverage, and cost bookkeeping without making retrieval calls. Do not use `ranks: [null, null]` for an unexecuted probe: that claims both searches ran and missed the target. Unknown fields, duplicate JSON object keys, nonstandard NaN/Infinity values, invalid ranks, and duplicate IDs are rejected. Invalid records are excluded from all other summaries; their costs are not silently counted as zero.

Generic outcomes can represent a caller-defined probe endpoint, for example:

```json
"outcomes": {"values": [0.2, 0.8], "bounds": [0, 1], "higher_is_better": true}
```

Do not fill these values with future-task relevance labels. The known target is part of the probe construction, not an evaluation query's relevance judgment. The tool does not load task labels or inspect their provenance.

Cost records **paired marginal costs**, using a unit you define consistently. Zero is legitimate for planned probes or cache hits, but the report cannot tell those apart. Record planned costs in a separate preflight log if useful, then audit a log of actual costs after execution. Keep sampling, generation, setup, storage, and other costs in a separate ledger unless explicitly allocated to the two sides. This audit cannot reconstruct a total experiment budget from paired search costs alone. Different units are reported separately; cost categories overlap and must not be added together.

## Read the diagnostics

| Finding | What it establishes | Next step |
| --- | --- | --- |
| Exact query collision | Query strings are byte-for-byte equal. Under deterministic retrieval and equal non-query settings, this pair cannot identify a query-transformation effect. | Regenerate the pair, or use it explicitly as a repeated-query control. |
| Token-bag collision | Queries have the same Unicode-normalized, case-folded token multiset. | Check the backend analyzer. Phrase syntax and semantic search may still distinguish the pair. |
| Zero observed contrast | Known-target ranks match, or numeric endpoints differ by at most `1e-12`. | Inspect probe difficulty, endpoint coarseness, and stochastic variation. Equal endpoints do not imply equal result lists. |
| Saturated success | Both known targets rank first, or both outcomes exactly reach the declared best bound. | Add indirect or harder probes; preserve a range of difficulties. |
| Observed floor | Both targets are absent at cutoff, or both outcomes exactly reach the worst bound. | Check the target, query difficulty, and retrieval depth. A censored miss does not reveal deeper rank. |
| Same query, different outcomes | Equal strings produced different recorded endpoints. | Check randomness, index updates, non-query settings, and logging order. |
| Sparse coverage | Some family/action-pair/bucket cells contain few valid probes or few distinct query pairs. | Declare the expected grid and add relevant missing cases. |

Normalization is `Unicode NFKC → casefold → Unicode \w+ tokens → sort`, preserving repetitions. It is a deliberately simple heuristic, not a simulation of any particular search service's tokenizer.

The coverage grid crosses observed families with the union of observed and explicitly expected action pairs and buckets. Action-pair order does not matter. The default minimum of 3 is an editable warning threshold, **not a sample-size recommendation or power calculation**. Inspect whether the crossed grid is meaningful for your design. Entirely unobserved, undeclared families cannot be detected.

```sh
searchprobe audit probes.jsonl --output report.json \
  --expect-pair original rewrite --expect-pair original expand \
  --expect-bucket direct --expect-bucket indirect --min-per-cell 8
```

Reports include counts and rates with explicit denominators, per-cell coverage, costs by unit, field-addressed validation errors, actionable diagnostics, and limits of inference. Rates without observations are `null`, not zero. Example diagnostics retain at most five probe IDs; the report does not copy query strings or known-target IDs.

## Call from Python before retrieval

```python
from searchprobe import audit_records, canonical_query, validate_record

probe = {
    "probe_id": "planned-01", "family": "indirect",
    "actions": ["original", "rewrite"],
    "queries": ["what do I owe", "income tax liability"],
    "cost": {"values": [0, 0], "total": 0, "unit": "search_calls"},
}
assert validate_record(probe) == []
report = audit_records([probe], min_per_cell=1)
assert report["counts"]["exact_distinct_query_pairs"] == 1
# A structural pass means this experiment is possible, not that it will help.
```

`audit_records(records, *, expected_action_pairs=None, expected_buckets=None, min_per_cell=3)` accepts an iterable and returns a JSON-safe dictionary. `audit_jsonl(path, **kwargs)` reads the same contract and retains malformed-line diagnostics. `validate_record(record)` returns field-addressed errors; cross-record duplicate IDs are checked by the audit. `canonical_query(query)` returns the normalized token tuple. These probe-audit functions perform no retrieval, modify no records, read no task labels, and call no model. The optional source-label pathway lives in the separate `searchprobe.decisions` module.

To inspect the first lexical pilot's **actual saved generator examples**, the adapter below uses only the small example sample already in `probe_audit.json`. It does not claim full-pilot collision rates, fabricate missing queries, or add retrieval outcomes. Full historical profile logs do not contain query strings, so they cannot be faithfully adapted to this contract without rerunning generation.

```sh
PYTHONPATH=packages/searchprobe/src python3 packages/searchprobe/examples/audit_lexical_examples.py \
  papers/learning-to-explore-search-environments/pilot/results/probe_audit.json \
  --output /tmp/searchprobe-lexical-examples.json
```

Run the dependency-free tests from the repository root:

```sh
PYTHONPATH=packages/searchprobe/src python3 -m unittest discover -s packages/searchprobe/tests -v
```

The package's GitHub Actions workflow runs the standard-library tests on Python 3.10, 3.12, and 3.14, and builds and installs a wheel for CLI smoke tests on Python 3.12. It has no publication step.

## Audit the semantic experiment's exported traces

After retrieval completes, run:

```sh
python3 papers/learning-to-explore-search-environments/semantic_pilot/audit_trace_exports.py
```

The frozen experiment saves raw action-pair renderings in `semantic_pilot/results/probe_traces/` and raw diagnostics in `probe_diagnostics/`. Decomposed actions may fuse multiple requests, so their newline-joined query renderings do not satisfy this tool's single-query contract. The exporter preserves both raw directories and creates `strict_probe_traces/`, `strict_probe_diagnostics/`, and `trace_provenance.json` under the same results directory. It runs the CLI on each strict file.

Strict exports retain original-vs-keywords/semantic/hyde pairs, with query text and ranks unchanged, and disclose excluded decomposed pairs. Their cost unit is `nominal_selected_probe_search_calls`: these are per-pair selection costs, not actual harness requests or total experiment spend. The provenance file explains source hashes, filtering, shared reference queries, caching, and the separate budget ledger. Raw traces already omit invalid and identical generated pairs, so their collision rates describe admitted candidates rather than the full generator.

The newly authored code and documentation in `packages/searchprobe` are available under the [MIT license](LICENSE). This package-scoped license does not relicense other repository files, dependencies, or reference material.
