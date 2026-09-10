# Full-paper research

Start with the [technical blog draft](searchprobe-blog.html). It explains the idea, the controller failure, complementary evidence, and the new retrieval results. It is self-contained and works offline. This directory continues milestone `e330622` on `codex/searchprobe-full-paper`; the earlier protocols and evidence are preserved.

The full paper is **not finished**. The new experiments sharpen the problem, but do not establish a novel acquisition method that improves unseen search environments. The contribution we want to earn is a source-trained relationship between cheap, automatically checkable search experiments and the marginal value of later search decisions. Generic value of information, decision-region discovery, workload robustness, and lookahead are established ideas; [the literature audit](theory/LITERATURE.md) records close predecessors.

## Evidence from this cycle

| Study | What it establishes | What it does not establish |
| --- | --- | --- |
| [Controlled suite](controlled/RESULTS.md), 1,658 exact rows in 38 conditions | Immediate-value rewards can miss complementary evidence; adaptive probing can beat the best fixed set in a gated example; hidden workload shifts and misspecified observation models can reverse gains. | Retrieval effectiveness, novel VOI theory, or a universal ordering of methods. Tie sensitivities are included. |
| [Finite-channel retrieval](natural/README.md), 1,620 run rows | Two-step acquisition .3824 macro nDCG vs random .3782 and IG .3805, but below its own prior .3842 and original .3957. | A successful development gate or generalization to a fresh family. Backend identity is hidden, unlike the earlier milestone. |
| [Capacity audit](natural/capacity_audit.json) | Even the best four-bucket/five-action policy is below all-action RRF in five of six environments. A verified posterior witness and a weak-tie upper relaxation separate attainable from optimistic scores. | A bound on richer query-dependent policies or fusion. Floating LP statuses are checked; these are not exact rational infeasibility proofs. |
| [Fusion frontier](fusion_frontier/README.md) | Source-selected two-search fusion .4094 vs all-action fusion .4212 at six searches, both with one generation call and no onboarding. | Equal quality, a new fusion algorithm, or a probe-selection advantage. All operating points are retained. |
| [Query-level transfer](query_transfer/README.md) | Tests source-selected nearest-neighbor routing against the reconstructed original ridge/fixed pipeline, with no onboarding. | An unseen-family confirmation: these three corpora were already development data. |
| [Supporting theory](theory/THEORY.md) | Exact small workload games, a sharp restricted coverage/information tradeoff, and a counterexample to changing the order of worst case and expectation. | Novelty for standard minimax, DRD, perturbation bounds, or reward-free exploration. |

All retrieval summaries use the same 365 family-specific query IDs, evaluated with two correlated backends: 730 query/backend cases. There are **three development families**, not six independent domains. Source utility fits and target evaluation stay separate in code, but all three families have informed research decisions. No untouched-family confirmation claim is made.

## Run the useful part in five minutes

The [SearchProbe package](../../../packages/searchprobe/README.md) contains editable synthetic examples in its installed wheel. Run these commands from the repository root:

```sh
python -m pip install ./packages/searchprobe
searchprobe-demo --output-dir editable-examples
```

The [portable evidence replay](replay/README.md) recalculates 810 saved milestone run summaries, 600 paired-contrast means, and 12 conditional model certificates from 34 checksum-pinned files. It needs only Python and SearchProbe; no GPU, MLX, NumPy, model downloads, or ignored caches. It is analysis replay, not retrieval/training regeneration.

The new natural/fusion/query-transfer experiments also commit their minimized input evidence. Their READMEs distinguish portable reanalysis from the larger process that originally generated the retrieval data. The controlled and workload examples use only the standard library; the channel diagnostics use NumPy and SciPy.

## What determines the next paper

The failure now sits in the utility model and its transfer across collections, not simply in selector responsiveness. A four-bucket model leaves most per-query oracle headroom unreachable; source-selected query-aware models also need to beat strong fixed/fusion policies. More lookahead by itself is therefore not a credible contribution.

The next mechanism should estimate **how a change in observable search behavior changes action or fusion utility while holding the corpus and relevance judgments fixed**. Within-corpus search interventions can separate those responses from collection fingerprints. The [frozen adapter and endpoint audit](interventions/README.md) prepare this test without inspecting interior-mixture outcomes. Any learned relationship must then predict effects in another family, with grouped evaluation that does not count intervention variants as independent domains. This is a research hypothesis, not an established causal guarantee or novelty claim.

A full-paper claim requires a frozen source model, a useful calibrated error/capacity diagnostic, strong source-selected fusion and query-aware baselines, complete onboarding/serving costs, and untouched collection families. A positive engineered example or a gain over a weak starting router will not meet that bar. Keep the current three families for development; evaluate fresh families only after the mechanism earns that expense and interpretive cost.
