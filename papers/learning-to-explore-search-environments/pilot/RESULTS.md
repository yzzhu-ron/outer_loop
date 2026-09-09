# First pilot: revise the action and probe library before learning a selector

**9 September 2026 · exploratory results, not a confirmatory study.**
The first feasibility stage is complete. The four lexical actions leave less
than 0.05 nDCG@10 oracle headroom in every tested environment, and the passive
profile correction changes zero task actions. This pilot does not justify
building a learned selector for the current library. It does not reject the
broader proposal, which calls for stronger semantic rewrites and indirect probes.

## What was run

The full SciFact corpus (5,183 documents) and FiQA corpus (57,638 documents), each
with BM25 and pinned `all-MiniLM-L6-v2` dense retrieval. All documents remain in
each index; MiniLM represents only the first 256 tokens of each document.
The duplicate-group split retained 249/300 SciFact and 402/648 FiQA test queries;
150 fixed queries from each family were evaluated under both backends.
These are 300 distinct test queries, not 600 independent tasks or four
independent corpus families.

Each transfer direction trains the router on 192 queries from the other corpus
and calibrates the profile correction on 64 separate source query IDs. Target
onboarding uses no real queries or labels. Three seeds, two probe families, and
operation caps of 0/16/64/256 share the same samples and action menu. The largest
body-probe history inspects 64 documents and performs 128 searches: 192 actual
operations under a cap of 256. This is fixed-coverage probing, not learned selection.

## Action-selection ceiling

| Environment | Original | Source router | Best fixed oracle | Per-query oracle | Oracle − best fixed |
|---|---:|---:|---:|---:|---:|
| SciFact · BM25 | 0.6061 | 0.6209 | 0.6209 | 0.6488 | 0.0278 |
| SciFact · MiniLM | 0.6436 | 0.6487 | 0.6474 | 0.6883 | 0.0409 |
| FiQA · BM25 | 0.2525 | 0.2523 | 0.2525 | 0.2783 | 0.0258 |
| FiQA · MiniLM | 0.3921 | 0.3847 | 0.3921 | 0.4297 | 0.0376 |

Both oracles use target relevance labels and are diagnostic only. Their gaps
over the source router range from 0.0260 to 0.0450. Neither comparison meets
the proposal's provisional 0.05 screen in any environment. This is a practical
screen for this limited library, not a significance test or a universal ceiling.

## Why the probe signal is insufficient

The body-term generator passes its mechanical checks but produces weak
experiments. In every history, 43/64 assigned action pairs are identical strings:
keyword filtering and lead-clause extraction do nothing to its short term lists.
Only the 21 identifier-trimming pairs can differ. Almost every probe has 12 terms,
so feature-bucket coverage is narrow. Example probes and counts are preserved in
[`results/probe_audit.json`](results/probe_audit.json).

BM25 retrieves the known document at rank 1 for **every body probe and both
actions**, across all seeds and both corpora. Its perfect synthetic score gives
no action-order information even though real-query actions differ. Dense body
probes have mean original-query reciprocal ranks of 0.4553 on SciFact and 0.4879
on FiQA; their few nonzero contrasts still produce no changed routing decisions.
The profile router's nDCG gain is exactly zero at every tested budget and seed.
The naive probe-only choice is less stable: at the 64-operation cap its mean
dense nDCG falls to 0.6087 on SciFact and 0.3623 on FiQA, below the respective
original-query scores of 0.6436 and 0.3921.

FiQA has no titles, so its exact-title probes are invalid. The title-profile
correction is consequently null in both transfer directions: either source
features or the target profile are empty. This is an unavailable/null control,
not evidence that informative title probes cannot transfer. Known-document ranks
also do not establish that the other returned documents are irrelevant.

## Cost and research decision

The full body profile spends 192 extra operations for zero gain. At 50 future
tasks this is 242 operations, versus 50 for the source router and 100 for the
two-search reciprocal-rank-fusion comparison. Every profile method still issues
one search per task, so no positive task-cost saving supports a break-even claim.
The plots show operation counts, not matched measured latency or dollar costs.
No paid API or LLM generation was used. Initial index/cache construction took
about 240 seconds locally; indexing/model loading dominated. Source fitting,
phase-specific search work, generation, and profile processing are logged separately.

**Gate A: revise** the action library; **Gate B: not established** for these
proxies; **Gate C: not tested**, because training a selector is premature;
**Gate D: no operational benefit** for the evaluated profile router.

The next experiment should freeze a stronger semantic rewrite and decomposition
library, audit action diversity before searching, then generate coherent indirect
probes that produce nontrivial paired contrasts. Repeat the ceiling and signal
tests before selecting or tuning a probe-value learner. Keep SciFact and FiQA
as pilot-only families. Entity/near-duplicate separation, profile transplantation,
strong active-selection baselines, and end-to-end task evidence remain outstanding.

## Evidence and verification

[Headroom](results/figures/headroom.png) · [Probe alignment](results/figures/alignment.png)
· [Quality versus operation cost](results/figures/adaptation_cost.png).
The CSVs, per-query outcomes/recall, numeric profiles, source model coefficients,
and conditional intervals are in [`results/`](results/). Three histories and two
families do not support population-level confidence claims. No selected test
query has blank support; one selected FiQA source query does. Missing judgments
and dense truncation remain limitations.

The 24 new tests and 3 existing tests pass. A cached validation rerun reproduced
every test-action score exactly under one enforced code/protocol contract.
Initial cold-cache timing is preserved separately: protocol annotations were
finalized during that first build, without changes to the engine or task scores.
This rerun is verification, not another scientific replicate.
See the [protocol](PROTOCOL.md) and [reproduction instructions](../README.md).
