# Search-environment exploration: frozen first pilot

Protocol v0.1.0, 9 September 2026. This protocol is recorded before inspecting
retrieval outcomes. The machine-readable defaults are in `protocol.json`.
The parent proposal is commit `a24a29a`. Any later change to these defaults must
be recorded as a dated amendment, including whether relevant outcomes had
already been inspected.

This exploratory pilot asks whether the fixed action library has useful
selection headroom and whether passive document-rediscovery profiles predict
real-query action utility. It does not train a probe selector. SciFact and FiQA
become pilot-only families once their results inform method choices; neither
can subsequently be described as an untouched final target.

## Data and permitted information

Read the released SciFact and FiQA BEIR ZIP archives directly, without
extracting archive paths. `protocol.json` pins the download URLs and archive
SHA-256 digests. The index contains every released corpus document. Index
construction is an offline backend operation, separately costed; onboarding
cannot rebuild or modify that index.

The retrieval configurations are sparse BM25 and dense
`sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Dense inputs are truncated to
256 model tokens and embeddings are normalized. A released document remains
the retrieval unit; truncation is not represented as document removal. Both
routers and baselines receive the accurate interface and backend identity.
Configurations sharing a corpus are correlated observations.

The evaluator can read qrels and the complete corpus to establish availability,
splits, and scores. The target runtime can inspect only documents selected
from its exploration-seed pool. Query/probe generation receives the sampled
document; profile construction receives numeric pairwise outcomes and query
buckets. No target task query, qrel, or eligibility decision is available
during onboarding. Backend corpus statistics are not exposed as generator or
router features. Returned passages and document IDs are absent from the
persisted profile. The profile is frozen during evaluation, and task-specific
state is reset between questions.

## Split and sampling rules

`make_split` groups documents by NFKC-normalized, casefolded, whitespace-
collapsed **body text**, so changing a title does not separate an otherwise
identical document. Empty bodies fall back to similarly normalized titles.
The content is prefixed with `body` or `title` and a NUL before SHA-256 hashing.
A second hash of `20260909:GROUP_HASH` assigns each duplicate group to the
exploration pool with probability 0.2. The remaining groups form the held-out
pool. Assignment uses no query text or qrels and is independent of input order.
The realized document fraction can differ from 20%, because groups stay intact.

For each original qrels split, retain a query only when it has positive support,
every judged document exists in the full corpus, and **all** positively judged
support is in the held-out pool. Nonpositive judgments may reference the
exploration pool. Missing corpus references invalidate the query and are
reported. The audit separately counts mixed support, exploration-only support,
empty positive support, missing references, duplicate groups, and query
retention. All documents remain indexed. This is document/normalized-duplicate
separation; entity disjointness and near-duplicate separation are not established.

Sort eligible query IDs by the runner's SHA-256 of
`json.dumps([salt, str(query_id)], sort_keys=True).encode()`, using Python's
default JSON separators. The salt is `test-20260909` for target test queries
and `train-20260909` for source train queries. Use at most 150 queries from
each target's released `test` qrels and
at most 256 from each source's released `train` qrels. The first
`min(192, max(1, floor(0.75 * selected_source_count)))` source IDs train the router;
the remaining IDs calibrate profile corrections. At 256 queries this is a
192/64 split.
These source query IDs are disjoint, but their support documents need not be.
Every method and backend uses the same selected real queries.

Run both source/target directions: SciFact to FiQA and FiQA to SciFact. All
configurations of a family remain in its assigned role. The two directions
are exploratory role rotations. There is no independent development family;
the defaults below are fixed rather than selected using target labels.

## Fixed actions and passive profiles

The common action menu is `original`, `keywords`, `identifier_terms`, and
`lead_clause`, implemented as deterministic lexical transformations. There
are no LLM calls. The result manifest records their exact implementations and
empty/identical-action rates. These actions do not include a strong semantic
or reasoning rewrite. A failed headroom screen therefore rejects only this
reduced library on these tested configurations.

Probe families are `exact_title`, an overlap/easy control with no empty-title
fallback, and `body_terms`, a lexical document-derived proxy. Empty titles
produce invalid probes: charge one document inspection, perform no searches,
and leave the profile unchanged. FiQA's absent titles make this control
uninformative there; preserve and report that failure.
`body_terms` is not a validated indirect semantic query. Audit copied spans,
degenerate transformations, empty titles, and query validity. Known-document
reciprocal rank does not establish that other retrieved documents are irrelevant.

For onboarding seeds 11, 23, and 47, sort exploration document IDs
lexicographically, shuffle using Python `random.Random(seed).shuffle`, and
take at most the first 64. Use that ordered sample across methods, backends,
and probe families within the corpus. Inspect a
candidate only when selected; there is no free preinspected candidate pool.
Cycle the paired comparison between original and each of the other three
actions. A selected pair costs one document inspection and two searches.
Budgets of 0, 16, 64, and 256 operations therefore allow at most 0, 5, 21,
and 64 completed valid pairs. An invalid probe consumes its document
inspection but no searches; its reason and actual costs are recorded.
the largest budget spends at most 192 operations. Report this underspend.
Cached unselected outcomes must remain inaccessible to any runtime policy.

For each action relative to original, store global and query-bucket means of
probe reciprocal-rank differences, shrunk by `n / (n + 3)`. The persisted
profile contains only numeric counts and differences. Probe families are
separate conditions. These passive histories do not establish an advantage
from active or learned probe selection.

The frozen lexical bucket rule is
`2 * int(token_count > 8) + int(any_identifier)`, giving buckets 0 through 3
using the engine's tokenizer and identifier predicate. The threshold was
fixed at eight before retrieval results were inspected, so body-term probes
with a maximum of 12 tokens can populate both length categories.

## Source training and comparisons

Train a multioutput Ridge router with `alpha=10` on the 192 source router-
training queries, using deployment-computable query features and a documented
backend flag. Predict action-specific nDCG@10. Within each backend, choose
the source fixed action using only those same source training outcomes.

Freeze that router. On the disjoint source calibration queries, fit a
per-action Ridge correction with `alpha=10`, `fit_intercept=False`, using
source histories across the predefined seeds and budgets. Its two inputs
are shrunk global and query-bucket action-minus-original probe deltas. Its
target is the real action-minus-original nDCG difference minus the frozen
router's predicted difference. At target time add the frozen correction to
the source router; no target labels fit, select, or update its weights.

Report identity, the source fixed action, the source router, probe-only
routing, and the source-calibrated passive profile correction. Probe-only
routing chooses the largest shrunk global probe delta, with original fixed
at zero. Include a two-search inference baseline that fuses original and
the best non-original alternative selected on source training data using
reciprocal-rank fusion. Charge both searches. This baseline is not necessarily
matched to every onboarding budget/workload point.

Separately report the target-best fixed-action oracle and per-query
best-action oracle. These use target labels and are ineligible deployed
methods. Compare oracle headroom against both the source router and the
target-best fixed action, rather than only identity. The 0.05 absolute
nDCG@10 screen is a descriptive opportunity threshold, not a significance test.

## Outcomes, costs, and decisions

The primary diagnostic is real-query nDCG@10, reported for every corpus and
backend and as an equal-family macro average. Synthetic reciprocal rank is a
different outcome. Examine pairwise action-order agreement, action regret,
and actual profile-induced real-task gains; correlation alone is insufficient.
Use paired queries and onboarding seeds. Shared profiles/histories are
adaptation units, and the two families cannot support broad population claims.

Report onboarding and serving costs separately for workloads 1, 10, 50, and
200, with 50 as the reference workload. Count sampled documents, both searches
per probe, generation/profile-processing time, and every task search. LLM
generation tokens are zero in this lexical implementation. Record measured
CPU latency, and separate offline indexing, source fitting, and cache-building
compute. Hidden experimental caches do not make target interactions free.
Only report a simple break-even workload when quality is comparable and
per-task cost savings are positive; otherwise show quality–cost curves.

Gate A asks whether more than one natural configuration has meaningful
headroom in this library. Gate B asks whether profiles help disjoint real
queries beyond source priors and lexical controls. Gate C remains untested
until a learned selector is compared with matched random, fixed-coverage,
and uncertainty selection. Gate D remains descriptive until quality at
matched total cost and stronger uses of inference-time compute are tested.

No result here establishes semantic-probe validity, entity-disjoint transfer,
end-to-end answer quality, autonomous-agent improvement, untouched-family
generalization, or the intended main-track contribution. A negative or
mixed outcome should drive a recorded revision of the action/probe design,
not a stronger claim from the same pilot data.
